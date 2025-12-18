from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCamResult:
    cam: torch.Tensor
    score: torch.Tensor


def _find_module(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise ValueError(f"Cannot find module named '{name}'")


def normalize_heatmap(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize to [0, 1] per-sample."""
    if cam.dim() != 4:
        raise ValueError(f"Expected cam as [B,1,H,W], got shape={tuple(cam.shape)}")
    b = cam.size(0)
    cam_ = cam.view(b, -1)
    cam_min = cam_.min(dim=1, keepdim=True).values
    cam_max = cam_.max(dim=1, keepdim=True).values
    cam_ = (cam_ - cam_min) / (cam_max - cam_min + eps)
    return cam_.view_as(cam)


class GradCAM:
    """Minimal Grad-CAM implementation.

    Works for any model where target_layer produces activations shaped [B,C,H,W].

    Typical usage:
        cam = GradCAM(model, target_layer="visual.layer4")
        out = cam(input_tensor, class_idx=pred)

    Notes for CLIP:
      - For ViT backbones, activations are usually [B, tokens, dim], not [B,C,H,W].
        This implementation targets conv feature maps (e.g., RN50). For ViT, prefer an
        attention-based rollout or reshape-token CAM variant.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device

        if isinstance(target_layer, str):
            self.target_layer = _find_module(model, target_layer)
        else:
            self.target_layer = target_layer

        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        def fwd_hook(_m, _inp, out):
            self._acts = out

        def bwd_hook(_m, _gin, gout):
            self._grads = gout[0]

        self._h1 = self.target_layer.register_forward_hook(fwd_hook)
        self._h2 = self.target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self._h1.remove()
        self._h2.remove()

    @torch.no_grad()
    def _check_acts(self):
        if self._acts is None:
            raise RuntimeError("No activations captured. Did you call forward?")
        if self._acts.dim() != 4:
            raise ValueError(
                f"Target layer output must be 4D [B,C,H,W] for Grad-CAM. Got {tuple(self._acts.shape)}"
            )

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[Union[int, torch.Tensor]] = None,
        forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> GradCamResult:
        """Compute CAM.

        Args:
            x: input tensor [B,3,H,W]
            class_idx: int or LongTensor [B]
            forward_fn: Optional wrapper to produce logits from x.

        Returns:
            GradCamResult(cam=[B,1,h,w] normalized (not resized), score=[B])
        """
        if self.device is not None:
            x = x.to(self.device)

        self.model.zero_grad(set_to_none=True)

        if forward_fn is not None:
            logits = forward_fn(x)
        else:
            # Thay vì gọi model(x) gây lỗi thiếu text, ta gọi encode_image
            import torch
            # Tạo tham số giả ind_prompt=0 để khớp với model PMG-AFT
            dummy_prompt = torch.tensor([0]).to(x.device)
            logits = self.model.encode_image(x, dummy_prompt)
        if logits.dim() != 2:
            raise ValueError(f"Expected logits [B,K], got {tuple(logits.shape)}")

        b, k = logits.shape

        if class_idx is None:
            cls = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            cls = torch.full((b,), class_idx, dtype=torch.long, device=logits.device)
        else:
            cls = class_idx.to(device=logits.device, dtype=torch.long)
            if cls.numel() != b:
                raise ValueError(f"class_idx must have B elements, got {cls.numel()} vs B={b}")

        scores = logits[torch.arange(b, device=logits.device), cls]

        score_sum = scores.sum()
        score_sum.backward(retain_graph=False)

        self._check_acts()
        assert self._grads is not None

        acts = self._acts
        grads = self._grads

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = normalize_heatmap(cam)
        return GradCamResult(cam=cam.detach(), score=scores.detach())


def cosine_similarity_map(
    feat_map: torch.Tensor,
    text_feat: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity per spatial location.

    Args:
        feat_map: [B,C,H,W] float tensor
        text_feat: [B,C] or [C]

    Returns:
        sim_map: [B,1,H,W] normalized to [0,1]
    """
    if feat_map.dim() != 4:
        raise ValueError(f"feat_map must be [B,C,H,W], got {tuple(feat_map.shape)}")
    b, c, h, w = feat_map.shape

    if text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0).expand(b, -1)
    if text_feat.dim() != 2 or text_feat.shape[1] != c:
        raise ValueError(f"text_feat must be [B,C] or [C] with C={c}, got {tuple(text_feat.shape)}")

    fmap = feat_map.view(b, c, h * w).transpose(1, 2)
    fmap = fmap / (fmap.norm(dim=-1, keepdim=True) + eps)

    t = text_feat / (text_feat.norm(dim=-1, keepdim=True) + eps)
    sim = (fmap * t.unsqueeze(1)).sum(dim=-1)

    sim = sim.view(b, 1, h, w)
    sim = normalize_heatmap(sim)
    return sim


def resize_cam(cam: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if cam.dim() != 4:
        raise ValueError("cam must be [B,1,h,w]")
    return F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
