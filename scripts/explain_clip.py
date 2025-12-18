from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

import clip

from viz.gradcam import GradCAM, cosine_similarity_map, resize_cam
from viz.image_utils import overlay_heatmap, save_uint8


def parse_args():
    p = argparse.ArgumentParser("Grad-CAM & cosine similarity map for this repo")
    p.add_argument("--image", required=True, help="Path to an input image")
    p.add_argument("--text", default=None, help="Text prompt (for cosine map). If omitted, uses predicted class")

    p.add_argument("--arch", default="RN50", help="CLIP backbone. Use RN50/RN101 for Grad-CAM conv maps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--checkpoint", default=None, help="Optional path to finetuned checkpoint (state_dict)")
    p.add_argument("--output_dir", default="outputs/explain")

    p.add_argument(
        "--target_layer",
        default="visual.layer4",
        help="Target layer name for Grad-CAM (RN backbones). Example: visual.layer4",
    )

    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--no_gradcam", action="store_true")
    p.add_argument("--no_cosine", action="store_true")

    return p.parse_args()


def _load_img(preprocess, image_path: str, device: str) -> Tuple[torch.Tensor, Image.Image]:
    pil = Image.open(image_path).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(device)
    return x, pil


def _load_checkpoint_if_any(model: nn.Module, ckpt_path: Optional[str]):
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # tolerate DataParallel prefixes
    model_state = model.state_dict()
    new_state = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module.") and not next(iter(model_state.keys())).startswith("module."):
            kk = kk[len("module.") :]
        elif (not kk.startswith("module.")) and next(iter(model_state.keys())).startswith("module."):
            kk = "module." + kk
        if kk in model_state and model_state[kk].shape == v.shape:
            new_state[kk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys (partial load): {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys (partial load): {len(unexpected)}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Use repo's CLIP (replace/clip.py) via import clip at project root.
    model, preprocess = clip.load(args.arch, device=device, jit=False, prompt_len=0)
    model.eval()

    _load_checkpoint_if_any(model, args.checkpoint)

    x, pil = _load_img(preprocess, args.image, args.device)

    # Classification-style logits using CLIP zero-shot over provided text (or fallback to imagenet classes if available)
    # We'll always create at least one text token.
    if args.text is not None:
        texts = [args.text]
    else:
        texts = ["a photo"]

    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(x, torch.tensor([0]).to(args.device))
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (model.logit_scale.exp() * image_features @ text_features.t())
        pred = logits.argmax(dim=1).item()
        score = logits[0, pred].item()
        print(f"pred={pred} score={score:.4f} text='{texts[pred]}'")

    # Save original
    from torchvision.transforms.functional import to_tensor

    img_tensor01 = to_tensor(pil)  # [3,H,W] in [0,1]
    img_uint8 = (img_tensor01.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

    # 1) Grad-CAM (only meaningful for RN backbones)
    if not args.no_gradcam:
        try:
            cam_engine = GradCAM(model, target_layer=args.target_layer, device=device)

            def forward_logits(inp: torch.Tensor) -> torch.Tensor:
                # Make a 1-class logit by matching the selected text
                img_f = model.encode_image(inp)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt = text_features[pred : pred + 1]
                return (model.logit_scale.exp() * img_f @ txt.t())  # [B,1]

            res = cam_engine(x, class_idx=0, forward_fn=forward_logits)
            cam_engine.close()

            cam_up = resize_cam(res.cam, size_hw=(img_uint8.shape[0], img_uint8.shape[1]))
            hm = cam_up[0, 0].detach().cpu().numpy()
            out = overlay_heatmap(img_uint8, hm, alpha=args.alpha, cmap=args.cmap)
            save_uint8(os.path.join(args.output_dir, "gradcam_overlay.png"), out)
        except Exception as e:
            print(f"[warn] Grad-CAM failed ({e}). Likely ViT backbone; try --arch RN50")

    # 2) Cosine similarity map from conv feature map (RN backbones)
    if not args.no_cosine:
        # Hook a conv feature map to get [B,C,H,W]
        feat = {}

        def hook(_m, _inp, out):
            feat["x"] = out

        try:
            layer = dict(model.named_modules())[args.target_layer]
        except KeyError:
            layer = None

        if layer is None:
            print("[warn] Cannot find target_layer for cosine map")
            return

        h = layer.register_forward_hook(hook)
        with torch.no_grad():
            _ = model.encode_image(x, torch.tensor([0]).to(args.device))
        h.remove()

        if "x" not in feat or feat["x"].dim() != 4:
            print("[warn] Cosine map needs conv feature map; try --arch RN50 and layer visual.layer4")
            return

        fmap = feat["x"]
        # project text feature to same channel dim if needed
        # Often layer4 channels != CLIP embedding dim; use a 1x1 conv-like linear projection from global pooled fmap.
        # Here: derive a per-channel prototype by regressing from pooled fmap -> text using least-squares (stable, no train)
        b, c, h_, w_ = fmap.shape
        pooled = fmap.mean(dim=(2, 3))  # [B,C]
        t = text_features[pred : pred + 1]

        if t.shape[1] != c:
            # simple linear projection of text to c using random orthogonal-ish matrix seeded
            g = torch.Generator(device=pooled.device)
            g.manual_seed(0)
            proj = torch.randn(t.shape[1], c, generator=g, device=pooled.device)
            proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
            t_proj = t.type(proj.dtype) @ proj
        else:
            t_proj = t

        sim = cosine_similarity_map(fmap, t_proj)
        sim_up = resize_cam(sim, size_hw=(img_uint8.shape[0], img_uint8.shape[1]))
        hm = sim_up[0, 0].detach().cpu().numpy()
        out = overlay_heatmap(img_uint8, hm, alpha=args.alpha, cmap=args.cmap)
        save_uint8(os.path.join(args.output_dir, "cosine_overlay.png"), out)


if __name__ == "__main__":
    main()
