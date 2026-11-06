"""Serve trained SpatialVLA joint targets to the Isaac evaluator on localhost."""
import argparse
from pathlib import Path
import numpy as np
import torch
from websockets.sync.server import serve
from openpi_client import msgpack_numpy
from data import image_tensor
from adapter import JointSpatialVLA, load_checkpoint
from protocol import HORIZON, POLICY_TYPE, action_chunk


def validate_observation(obs):
    if set(obs) != {'state', 'top', 'wrist', 'prompt'}:
        raise ValueError('Unexpected policy input fields')
    if np.asarray(obs['state']).shape != (9,) or not np.isfinite(obs['state']).all():
        raise ValueError('Expected nine finite current joints')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip():
        raise ValueError('Missing task instruction')
    for key in ['top', 'wrist']:
        image = np.asarray(obs[key])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f'Invalid {key} RGB observation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8007)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.75)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    checkpoint = load_checkpoint(args.checkpoint)
    model = JointSpatialVLA(rank=checkpoint['config']['rank'], seed=checkpoint['config']['seed'])
    model.load_adapter(checkpoint['adapter'])
    model.eval()
    stats = checkpoint['stats']
    state_mean = np.asarray(stats['state']['mean'], dtype=np.float32)
    state_std = np.asarray(stats['state']['std'], dtype=np.float32)
    action_mean = np.asarray(stats['actions']['mean'], dtype=np.float32)
    action_std = np.asarray(stats['actions']['std'], dtype=np.float32)
    metadata = {'policy_type': POLICY_TYPE, 'checkpoint': str(args.checkpoint.resolve()),
                'horizon': HORIZON, 'action_dim': 9}

    def handle(socket):
        socket.send(msgpack_numpy.packb(metadata))
        try:
            for message in socket:
                obs = msgpack_numpy.unpackb(message)
                validate_observation(obs)
                images = torch.stack([image_tensor(obs[key]) for key in ['top', 'wrist']])[None].cuda()
                state = torch.tensor((obs['state'] - state_mean) / state_std)[None].cuda()
                with torch.inference_mode():
                    predicted = model(images, state, [obs['prompt']])
                targets = action_chunk(predicted[0].cpu().numpy() * action_std + action_mean)
                socket.send(msgpack_numpy.packb({'actions': targets}))
        except Exception as error:
            try:
                socket.send(str(error))
            finally:
                raise

    with serve(handle, '127.0.0.1', args.port, compression=None, max_size=8 * 1024**2) as server:
        print(f'SpatialVLA ready on localhost:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
