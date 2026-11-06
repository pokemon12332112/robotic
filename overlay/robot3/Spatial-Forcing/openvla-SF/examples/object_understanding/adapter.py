"""Spatial-Forcing adapted to nine absolute joint targets.

Uses the official bidirectional Llama fork, frozen OpenVLA visual features,
language Q/V LoRA, an explicit proprio token, and 8 x 9 zero action placeholders.
VGGT supervises layer-24 visual tokens with the upstream alignment projector.
The residual L1 head is narrowed to fit training on a single RTX 4090.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import MLPResNet
from prismatic.models.projectors import ProprioProjector, AlignProjector
from data import HORIZON

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / 'ckpts/openvla-7b'
FORMAT = 'spatial_forcing_object_understanding_joint_lora_v1'
ACTION_DIM = 9
ALIGN_LAYER = 24


def code_identity():
    digest = hashlib.sha256()
    files = [Path(__file__).with_name(n) for n in ['adapter.py', 'data.py', 'train.py', 'teacher.py']]
    files += sorted((ROOT / 'vggt').rglob('*.py'))
    files += [ROOT / 'prismatic/util/pooling_utils.py']
    files += sorted((ROOT / 'prismatic/extern/hf').glob('*.py'))
    files += [ROOT / 'prismatic/models/action_heads.py', ROOT / 'prismatic/models/projectors.py',
              Path(modeling_llama.__file__)]
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_identity(path=PRETRAINED):
    path = Path(path)
    teacher = ROOT / 'ckpts/vggt/model.pt'
    source = next(item for item in json.loads((ROOT.parent / 'local_setup/checkpoint-sources.json').read_text())
                  if item['repo'] == 'facebook/VGGT-1B')
    return {'source': json.loads((ROOT.parent / 'local_setup/base-checkpoint-source.json').read_text()),
            'teacher': {'source': source, 'bytes': teacher.stat().st_size,
                        'mtime_ns': teacher.stat().st_mtime_ns},
            'files': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in sorted(path.iterdir()) if p.suffix in {'.json', '.safetensors', '.model'}}}


def prepare_tokens(tokenizer, prompts, device):
    """No ground-truth actions enter the network: build only prompt/query/stop tokens."""
    sequences, lengths = [], []
    for prompt in prompts:
        ids = tokenizer(f'In: What action should the robot take to {prompt.lower()}?\nOut:',
                        add_special_tokens=True)['input_ids']
        if ids[-1] != 29871:
            ids.append(29871)  # Official inference convention: space after Out:
        if len(ids) > 256:
            raise ValueError('Instruction too long; refusing silent truncation')
        lengths.append(len(ids))
        sequences.append(ids + [1] * (HORIZON * ACTION_DIM) + [2])
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    queries = torch.zeros_like(ids, dtype=torch.bool)
    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, :len(sequence)] = 1
        queries[row, length:length + HORIZON * ACTION_DIM] = True
    return ids, attention, queries, torch.tensor(lengths, device=device)


class JointSF(nn.Module):
    def __init__(self, device='cuda', rank=8, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.device_name = str(device)
        dtype = torch.bfloat16 if self.device_name.startswith('cuda') else torch.float32
        print('Loading base OpenVLA for Spatial-Forcing with bidirectional SDPA', flush=True)
        self.backbone = OpenVLAForActionPrediction.from_pretrained(
            PRETRAINED, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation='sdpa').to(device)
        self.backbone.vision_backbone.set_num_images_in_input(2)
        self.backbone.requires_grad_(False)
        core = self.backbone.language_model.model
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0,
                                target_modules=['q_proj', 'v_proj'], bias='none'), core)
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        core.config.use_cache = False
        if not all(isinstance(layer.self_attn, modeling_llama.LlamaSdpaAttention) for layer in core.layers):
            raise ValueError('OFT requires the custom bidirectional SDPA attention implementation')
        hidden = self.backbone.config.text_config.hidden_size
        self.proprio = ProprioProjector(hidden, ACTION_DIM).to(device)
        self.align_projector = AlignProjector(hidden, 1024, 'cosine', use_vlm_norm=True).to(device)
        self.head = MLPResNet(num_blocks=2, input_dim=hidden * ACTION_DIM,
                             hidden_dim=512, output_dim=ACTION_DIM).to(device)
        nn.init.normal_(self.head.fc2.weight, std=0.001)
        nn.init.zeros_(self.head.fc2.bias)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, local_files_only=True)
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
            self.backbone.language_model.model.train(mode)
        return self

    def forward(self, images, state, prompts, *, return_spatial_features=False):
        b = images.shape[0]
        if images.shape != (b, 2, 6, 224, 224) or state.shape != (b, ACTION_DIM):
            raise ValueError('Expected DINO/SigLIP transforms of two views and nine joints')
        ids, attention, queries, lengths = prepare_tokens(self.tokenizer, prompts, state.device)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device_name.startswith('cuda') else contextlib.nullcontext()
        with autocast:
            with torch.no_grad():
                embeddings = self.backbone.get_input_embeddings()(ids)
                embeddings = embeddings.masked_fill(queries.unsqueeze(-1), 0)
                vision = self.backbone._process_vision_features(images.flatten(1, 2))
            visual_and_state = torch.cat([vision, self.proprio(state).unsqueeze(1)], dim=1)
            inputs, mask = self.backbone._build_multimodal_attention(embeddings, visual_and_state, attention)
            # Expose layer 24 only for teacher alignment; never compute vocabulary logits.
            output = self.backbone.language_model.model(
                inputs_embeds=inputs, attention_mask=mask, use_cache=False,
                output_attentions=False, output_hidden_states=return_spatial_features, return_dict=True)
            hidden = output.last_hidden_state
            # Match OFT's next-token alignment: the first output uses the last prompt token.
            indices = lengths[:, None] - 1 + visual_and_state.shape[1] + torch.arange(
                HORIZON * ACTION_DIM, device=state.device)[None]
            action_states = hidden[torch.arange(b, device=state.device)[:, None], indices]
            predicted = self.head(action_states.reshape(b, HORIZON, -1))
        if return_spatial_features:
            # BOS precedes two 256-token views; the proprio token is excluded.
            spatial = output.hidden_states[ALIGN_LAYER][:, 1:1 + vision.shape[1], :]
            return predicted.float(), spatial
        return predicted.float()

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
        raise ValueError('Expected a trained Spatial-Forcing joint-space checkpoint')
    if checkpoint['code_identity'] != code_identity():
        raise ValueError('Adapter implementation differs from the saved checkpoint')
    if checkpoint['pretrained'] != file_identity():
        raise ValueError('Pretrained OpenVLA weights differ from this checkpoint')
    if checkpoint['stats']['split'] != 'train' or checkpoint['stats']['horizon'] != HORIZON:
        raise ValueError('Invalid training normalization provenance')
    return checkpoint
