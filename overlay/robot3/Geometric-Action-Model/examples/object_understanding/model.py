"""GAM joint-space LoRA adaptation with frozen geometric backbone and T5.

One observed timestep; previous-action slots are consistently masked to zero.
Future geometric tokens remain supervised, while depth decoding is disabled.
"""
import contextlib
import hashlib
import inspect
import io
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from data import HORIZON
from robot.modeling.action_head_v2 import ActionHeadV2
from robot.modeling.conditioning import TextConditioner
from robot.modeling.da3_giant_encoder import DA3GiantEncoder
from robot.modeling.future_predictor import GAMFuturePredictor

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / 'checkpoints_hf/3da-libero-gam/pretrained/pretrained-gam.pt'
BASE = ROOT / 'checkpoints/track4world_da3.pth'
FORMAT = 'gam_object_understanding_lora_v1'


def file_identity(path):
    # HF downloads store the upstream content digest next to each file.
    path = Path(path).resolve()
    metadata = path.parent / '.unused'
    if path == PRETRAINED.resolve():
        metadata = ROOT / 'checkpoints_hf/3da-libero-gam/.cache/huggingface/download/pretrained/pretrained-gam.pt.metadata'
    elif path == BASE.resolve():
        metadata = ROOT / '.cache/huggingface/download/checkpoints/track4world_da3.pth.metadata'
    return {'path': str(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns,
            'hf_metadata': metadata.read_text().splitlines()[:2] if metadata.is_file() else None}


def code_identity():
    digest = hashlib.sha256()
    for name in ['data.py', 'model.py', 'train.py']:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def add_lora(module, rank):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
        else:
            add_lora(child, rank)


class JointGAM(nn.Module):
    def __init__(self, device='cuda', rank=8, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.device_name = str(device)
        self.rank = rank
        self.seed = seed
        self.dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        print('Loading pretrained GAM geometry and predictor', flush=True)
        ckpt = torch.load(PRETRAINED, map_location='cpu', mmap=True, weights_only=False)
        cfg = ckpt['config']
        self.encoder = DA3GiantEncoder(ckpt_path=str(BASE), freeze_backbone=True,
                                      n_action_steps=5, views_per_timestep=2,
                                      action_steps_per_token=HORIZON)
        self.encoder.load_state_dict(ckpt['student_da3'], strict=True)
        self.encoder.requires_grad_(False).eval().to(device=device, dtype=self.dtype)
        keys = inspect.signature(GAMFuturePredictor).parameters
        options = {k: v for k, v in cfg['predictor'].items() if k in keys}
        options.update(proprio_dim=9, action_dim=9, action_chunk_size=HORIZON,
                       gradient_checkpointing=True)
        self.predictor = GAMFuturePredictor(**options)
        state = ckpt['future_predictor']
        current = self.predictor.state_dict()
        mismatched = {k for k, v in state.items() if k in current and v.shape != current[k].shape}
        allowed = {'proprio_token_proj.0.weight', 'action_history_proj.0.weight',
                   'future_proprio_proj.weight', 'future_proprio_proj.bias'}
        if mismatched != allowed or set(state) != set(current):
            raise ValueError(f'Unexpected pretrained predictor layout: {mismatched}')
        self.predictor.load_state_dict({k: v for k, v in state.items() if k not in allowed}, strict=False)
        self.predictor.requires_grad_(False)
        add_lora(self.predictor.blocks, rank)
        # Joint angles are a different representation from pretrained EEF poses.
        self.predictor.proprio_token_proj[0].reset_parameters()
        self.predictor.proprio_token_proj.requires_grad_(True)
        self.predictor.action_history_proj[0].reset_parameters()
        self.predictor.action_history_proj.requires_grad_(True)
        self.head = ActionHeadV2(n_dims=9, chunk_size=HORIZON)
        self.head.load_state_dict({k: v for k, v in ckpt['action_head'].items()
                                   if k not in {'output.weight', 'output.bias'}}, strict=False)
        # Preserve the pretrained trunk, initialize the joint-space output.
        nn.init.normal_(self.head.output.weight, std=0.001)
        nn.init.zeros_(self.head.output.bias)
        self.predictor.to(device=device, dtype=self.dtype)
        self.head.to(device=device, dtype=self.dtype)
        # Keep optimizer-owned parameters fp32; autocast handles matrix products.
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        del ckpt
        self.text = TextConditioner(encoder_type='t5', cache_token_embeddings=True,
                                    cache_max_entries=1024).requires_grad_(False).eval().to(device)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'encoder'):
            self.encoder.eval()
        if hasattr(self, 'text'):
            self.text.eval()
        return self

    def autocast(self):
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if self.device_name.startswith('cuda') else contextlib.nullcontext())

    def forward(self, images, state, prompts, future_images=None):
        b = len(images)
        with self.autocast():
            with torch.no_grad():
                images = self.encoder.normalize_images(images)
                visual = self.encoder.encode_shallow_visual_slots(images, T=1, V=2)['visual_tokens']
                language = self.text.encode_tokens(prompts)
                target = None
                if future_images is not None:
                    target = self.encoder.encode_shallow_visual_slots(
                        self.encoder.normalize_images(future_images), T=1, V=2)['visual_tokens']
            prediction = self.predictor(past_visual_tokens=visual,
                                        proprio_history=state[:, None],
                                        past_action_history=state.new_zeros(b, 1, HORIZON, 9),
                                        lang_feats=language['last_hidden_state'],
                                        lang_padding_mask=language['attention_mask'])
            deep = self.encoder.propagate_shallow_with_actions_grad(
                prediction['predicted_next_visual_tokens'], prediction['predicted_action_tokens'],
                decode_visuals=False, gradient_checkpointing=self.training)
            actions = self.head(deep['action_tokens'].reshape(b, 1, 2, -1))[:, 0].float()
            feature_loss = None
            if target is not None:
                feature_loss = (prediction['predicted_next_visual_tokens'].float() - target.float()).abs().flatten(1).mean(1)
        return actions, feature_loss

    def adapter_state(self):
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}

    def load_adapter(self, state):
        expected = {name for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != expected:
            raise ValueError('Adapter parameter names do not match this model')
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)


def load_checkpoint(path):
    content = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    checkpoint['_loaded_sha256'] = hashlib.sha256(content).hexdigest()
    if checkpoint.get('format') != FORMAT or checkpoint.get('step', 0) < 1:
        raise ValueError('Expected a trained object-understanding GAM adapter checkpoint')
    if checkpoint['code_identity'] != code_identity():
        raise ValueError('Adapter implementation differs from checkpoint; restore the training code')
    if checkpoint['pretrained'] != file_identity(PRETRAINED) or checkpoint['base'] != file_identity(BASE):
        raise ValueError('Pretrained source weights differ from this checkpoint')
    if checkpoint['stats']['split'] != 'train' or checkpoint['stats']['horizon'] != HORIZON:
        raise ValueError('Invalid training normalization provenance')
    return checkpoint
