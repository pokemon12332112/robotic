"""SpatialVLA's pretrained 3D visual-language backbone with a joint-space head.

Retains ZoeDepth + egocentric 3D position encoding. Replaces the original
seven-dimensional spatial action-token output with continuous nine-joint chunks.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoTokenizer

from data import HORIZON
from model import SpatialVLAForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / 'checkpoints/spatialvla-4b-224-pt'
FORMAT = 'spatialvla_object_understanding_joint_lora_v1'
CALIBRATION = Path(__file__).with_name('calibration.json')


def code_identity():
    digest = hashlib.sha256()
    for path in [*[Path(__file__).with_name(n) for n in
                   ['adapter.py', 'data.py', 'train.py', 'calibration.json']],
                 *sorted((ROOT / 'model').glob('*.py'))]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_identity(path=PRETRAINED):
    path = Path(path).resolve()
    return {'source': json.loads((ROOT / 'checkpoint-source.json').read_text()),
            'files': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in sorted(path.iterdir()) if p.suffix in {'.safetensors', '.json'}}}


def camera_intrinsics():
    calibration = json.loads(CALIBRATION.read_text())
    matrices = np.asarray([calibration['intrinsics'][key] for key in ['top', 'wrist']], dtype=np.float32)
    matrices[:, :2, :] *= 224 / 256
    return torch.from_numpy(matrices)


def validate_scene_calibration(payload):
    expected = json.loads(CALIBRATION.read_text())['rig_settings']
    settings = payload['run_config']['task_settings']
    for key, value in expected.items():
        if settings[key] != value:
            raise ValueError(f'Scene camera rig differs from training calibration: {key}')
    if payload['scene']['randomization_samples']['camera']:
        raise ValueError('Camera randomization requires per-observation calibration')


class JointSpatialVLA(nn.Module):
    def __init__(self, device='cuda', rank=8, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.device_name = str(device)
        self.rank = rank
        dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        print('Loading pretrained SpatialVLA with egocentric 3D encoding', flush=True)
        self.backbone = SpatialVLAForConditionalGeneration.from_pretrained(
            PRETRAINED, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation='eager').to(device)
        if not self.backbone.config.use_vision_zoe:
            raise ValueError('SpatialVLA depth/3D encoding must be enabled')
        self.backbone.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0,
                                          target_modules=['q_proj', 'v_proj'], bias='none'),
                                self.backbone.language_model)
        self.backbone.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.backbone.config.use_cache = False
        self.backbone.language_model.config.use_cache = False
        hidden = self.backbone.config.text_config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hidden + 9), nn.Linear(hidden + 9, 512),
                                  nn.GELU(), nn.Linear(512, HORIZON * 9)).to(device)
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.zeros_(self.head[-1].bias)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, local_files_only=True)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = 'right'
        self.register_buffer('intrinsics', camera_intrinsics().to(device), persistent=False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            # Vision, depth, and 3D modules remain deterministic and frozen.
            self.backbone.eval()
            self.backbone.language_model.train(mode)
        return self

    def tokenize(self, prompts):
        # Same prefix convention as the official processor, with two images.
        strings = ['<image>' * 512 + self.tokenizer.bos_token + prompt + '\n' for prompt in prompts]
        inputs = self.tokenizer(strings, add_special_tokens=False, padding=True, return_tensors='pt')
        if not torch.all((inputs['input_ids'] == self.backbone.config.image_token_index).sum(1) == 512):
            raise ValueError('Each observation must contain exactly two 256-token images')
        if inputs['input_ids'].shape[1] > 1024:
            raise ValueError('Instruction is too long; refusing to truncate image/prompt tokens')
        return {key: value.to(self.device_name) for key, value in inputs.items()}

    def forward(self, images, state, prompts):
        b = images.shape[0]
        if images.shape != (b, 2, 3, 224, 224) or state.shape != (b, 9):
            raise ValueError('Expected two RGB views and nine current joints')
        inputs = self.tokenize(prompts)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device_name.startswith('cuda') else contextlib.nullcontext()
        with autocast:
            outputs = self.backbone(**inputs, pixel_values=images.flatten(0, 1),
                                    intrinsic=self.intrinsics.repeat(b, 1, 1), use_cache=False,
                                    output_hidden_states=True, num_logits_to_keep=1, return_dict=True)
            lengths = inputs['attention_mask'].sum(1) - 1
            context = outputs.hidden_states[-1][torch.arange(b, device=state.device), lengths]
            actions = self.head(torch.cat([context.float(), state.float()], dim=-1))
        return actions.reshape(b, HORIZON, 9).float()

    def adapter_state(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapter(self, values):
        parameters = dict(self.named_parameters())
        expected = {name for name, p in parameters.items() if p.requires_grad}
        if set(values) != expected:
            raise ValueError('Adapter parameters do not match this model')
        with torch.no_grad():
            for name, value in values.items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f'Invalid adapter tensor: {name}')
                parameters[name].copy_(value)


def load_checkpoint(path):
    content = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    checkpoint['_loaded_sha256'] = hashlib.sha256(content).hexdigest()
    if checkpoint.get('format') != FORMAT or checkpoint.get('step', 0) < 1:
        raise ValueError('Expected a trained SpatialVLA object-understanding checkpoint')
    if checkpoint['code_identity'] != code_identity():
        raise ValueError('Adapter implementation differs from the saved checkpoint')
    if checkpoint['pretrained'] != file_identity():
        raise ValueError('Pretrained SpatialVLA weights differ from this checkpoint')
    if checkpoint['stats']['split'] != 'train' or checkpoint['stats']['horizon'] != HORIZON:
        raise ValueError('Invalid training normalization provenance')
    return checkpoint
