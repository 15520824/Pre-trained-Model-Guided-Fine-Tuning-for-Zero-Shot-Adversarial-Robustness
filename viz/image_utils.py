from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def to_numpy_img01(x: torch.Tensor) -> np.ndarray:
    """Tensor [3,H,W] in [0,1] -> uint8 HWC."""
    if x.dim() != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(x.shape)}")
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return x


def apply_colormap(heatmap01_hw: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """Return uint8 HWC color map."""
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    colored = cm(heatmap01_hw)[..., :3]
    colored = (colored * 255.0).astype(np.uint8)
    return colored


def overlay_heatmap(
    image_uint8_hwc: np.ndarray,
    heatmap01_hw: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    if image_uint8_hwc.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if heatmap01_hw.ndim != 2:
        raise ValueError("heatmap must be [H,W]")

    hm = apply_colormap(heatmap01_hw, cmap=cmap).astype(np.float32)
    img = image_uint8_hwc.astype(np.float32)
    out = (1 - alpha) * img + alpha * hm
    return np.clip(out, 0, 255).astype(np.uint8)


def save_uint8(path: str, img_uint8_hwc: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(img_uint8_hwc).save(path)
