"""Fine-tune the pretrained OpenVLA-OFT on the original object-understanding train split."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from data import DATA_ROOT, RobotDataset, compute_stats
from adapter import FORMAT, PRETRAINED, ROOT, JointOFT, code_identity, file_identity, load_checkpoint


def save_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/object-understanding/train')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9)
    args = parser.parse_args()
    if min(args.max_steps, args.save_every, args.batch_size, args.rank) < 1 or args.lr <= 0:
        parser.error('Steps, batch size, rank, and learning rate must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid memory fraction')
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = output / 'latest.pt'
        if (output / 'configuration.json').exists() and not args.resume:
            raise FileExistsError('Output already contains a run; use --resume or a new --output')
        previous = load_checkpoint(latest) if args.resume else None
        stats = compute_stats(args.data_root)
        if previous and previous['stats'] != stats:
            raise ValueError('Training dataset or normalization changed')
        config = {'rank': args.rank, 'seed': args.seed, 'lr': args.lr,
                  'batch_size': args.batch_size}
        if previous and previous['config'] != config:
            raise ValueError('Resume settings differ from the saved training configuration')
        save_json(output / 'configuration.json', {**config, 'max_steps': args.max_steps,
                  'data_root': str(args.data_root.resolve()), 'stats': stats, 'code_identity': code_identity()})
        save_json(output / 'normalization.json', stats)
        dataset = RobotDataset(stats, args.data_root)
        model = JointOFT(args.device, rank=args.rank, seed=args.seed).train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
        generator = torch.Generator().manual_seed(args.seed)
        start = 0
        if previous:
            model.load_adapter(previous['adapter'])
            optimizer.load_state_dict(previous['optimizer'])
            generator.set_state(previous['sample_rng'])
            torch.set_rng_state(previous['torch_rng'])
            if args.device.startswith('cuda'):
                torch.cuda.set_rng_state(previous['cuda_rng'])
            start = previous['step']
        if start >= args.max_steps:
            print(f'Already at step {start}; increase --max-steps to continue', flush=True)
            return
        count = sum(p.numel() for p in parameters)
        print(f'Training {count:,} parameters on {len(dataset)} train frames', flush=True)
        status = {'stage': 'running', 'step': start, 'max_steps': args.max_steps,
                  'train_episodes': 584, 'test_episodes_excluded': 148, 'trainable_parameters': count}
        save_json(output / 'status.json', status)
        try:
            with (output / 'metrics.jsonl').open('a') as log:
                for step in range(start + 1, args.max_steps + 1):
                    began = time.monotonic()
                    ids = torch.randint(len(dataset), (args.batch_size,), generator=generator).tolist()
                    batch = default_collate([dataset[index] for index in ids])
                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            batch[key] = value.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model(batch['images'], batch['state'], batch['prompt'])
                    mask = batch['action_mask'].unsqueeze(-1)
                    action_loss = ((predicted - batch['actions']).abs() * mask).sum() / (mask.sum() * 9)
                    loss = action_loss
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                    optimizer.step()
                    record = {'step': step, 'loss': loss.item(), 'action_loss': action_loss.item(),
                              'grad_norm': float(grad),
                              'seconds': time.monotonic() - began,
                              'source_episodes': batch['source_episode']}
                    if args.device.startswith('cuda'):
                        record['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    if step == 1 or step % 10 == 0 or step == args.max_steps:
                        print(json.dumps(record), flush=True)
                        status.update(step=step, last_loss=record['loss'])
                        save_json(output / 'status.json', status)
                    if step % args.save_every == 0 or step == args.max_steps:
                        payload = {'format': FORMAT, 'step': step, 'adapter': model.adapter_state(),
                                   'optimizer': optimizer.state_dict(), 'config': config, 'stats': stats,
                                   'sample_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                                   'cuda_rng': torch.cuda.get_rng_state() if args.device.startswith('cuda') else None,
                                   'pretrained': file_identity(PRETRAINED),
                                   'code_identity': code_identity()}
                        temporary = output / 'latest.pt.tmp'
                        torch.save(payload, temporary)
                        temporary.replace(latest)
                        # Preserve each saved checkpoint for comparisons and recovery.
                        import os
                        step_path = output / f'step_{step:06d}.pt'
                        if not step_path.exists():
                            os.link(latest, step_path)
                status.update(stage='complete', step=args.max_steps, checkpoint=str(latest))
                save_json(output / 'status.json', status)
        except BaseException as error:
            status.update(stage='error', error=str(error))
            save_json(output / 'status.json', status)
            raise


if __name__ == '__main__':
    main()
