"""Frozen VGGT targets from the same two current RGB views seen by the policy.

This module is loaded only by training. No future frames, actions, object poses,
depth labels, or held-out data are supplied to the teacher.
"""
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_normed_images
from prismatic.util.pooling_utils import custom_pooling
from data import image_processor


class SpatialTeacher:
    def __init__(self, checkpoint, device='cuda'):
        if not str(device).startswith('cuda'):
            raise ValueError('Spatial-Forcing training requires CUDA')
        print('Loading frozen pretrained VGGT spatial teacher', flush=True)
        self.model = VGGT(enable_camera=False, enable_point=False, enable_depth=False,
                          enable_track=False, feature_only=True)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        mismatch = self.model.load_state_dict(weights, strict=False)
        if mismatch.missing_keys or any(not key.startswith(
                ('camera_head.', 'point_head.', 'depth_head.', 'track_head.'))
                for key in mismatch.unexpected_keys):
            raise ValueError(f'Unexpected VGGT checkpoint mismatch: {mismatch}')
        del weights
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images):
        if images.ndim != 5 or images.shape[1:] != (2, 6, 224, 224):
            raise ValueError('Expected two current DINO/SigLIP RGB views')
        pixels = preprocess_normed_images(images.flatten(1, 2), image_processor(), 2)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.model(pixels)
        features = output['features'][-1][:, :, output['patch_start_idx']:, :]
        height, width = pixels.shape[-2:]
        # Upstream pooling needs only the reference token count, not policy values.
        reference = images.new_empty(images.shape[0], 512, 1)
        target = custom_pooling(features, (height // 14, width // 14), (height, width),
                                reference, 'bilinear', True)
        if target.shape != (images.shape[0], 512, 2048) or not torch.isfinite(target).all():
            raise ValueError('Invalid VGGT spatial targets')
        return target.detach()
