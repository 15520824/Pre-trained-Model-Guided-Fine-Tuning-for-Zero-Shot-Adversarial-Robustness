import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import io
import os
from collections import OrderedDict
from torchvision import transforms
from typing import Union

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="🛡️ Zero-Shot Robustness Lab", layout="wide")

# --- HELPER: IMAGE NORMALIZATION ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

def pmg_img_preprocessing(x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    return normalize_imagenet(x)

def clamp(x: torch.Tensor, lower: Union[torch.Tensor, float], upper: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)

# --- CLASS GRADCAM ---
class ClipGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.handle_f = self.target_layer.register_forward_hook(self.save_activation)
        self.handle_b = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): 
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0]

    def __call__(self, image_input, text_features, device, dummy_prompt):
        self.model.zero_grad()
        img_emb = self.model.encode_image(pmg_img_preprocessing(image_input.float()), dummy_prompt)
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        score = (img_norm * text_features).sum()
        score.backward(retain_graph=True)
        
        if self.gradients is None: 
            return np.zeros((224, 224))

        grads = self.gradients
        acts = self.activations
        
        # Xử lý chiều tensor [L, B, D] -> [B, L, D]
        if len(grads.shape) == 3 and grads.shape[1] == 1 and grads.shape[0] > 1:
            grads = grads.permute(1, 0, 2)
            acts = acts.permute(1, 0, 2)
        
        # Xử lý ViT
        if len(grads.shape) == 3: 
            grads = grads[:, 1:, :] 
            acts = acts[:, 1:, :]
            n_patches = grads.shape[1]
            if n_patches == 0: return np.zeros((224, 224))
            grid_size = int(np.sqrt(n_patches))
            grads = grads.permute(0, 2, 1).reshape(1, -1, grid_size, grid_size)
            acts = acts.permute(0, 2, 1).reshape(1, -1, grid_size, grid_size)
        elif len(grads.shape) == 2:
            return np.zeros((224, 224))

        # Tính Grad-CAM
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bicubic', align_corners=False)
        
        cam_np = cam.squeeze().cpu().detach().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np

# --- VẼ HEATMAP (FIX LỖI NGƯỢC MÀU) ---
def overlay_heatmap(img_pil, mask):
    img_np = np.array(img_pil.resize((224, 224)))
    
    # 1. Chuẩn hóa mask về [0, 1]
    # Thêm epsilon để tránh chia cho 0
    if mask.max() - mask.min() > 1e-8:
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    else:
        mask = np.zeros_like(mask)
    
    # 2. Tạo Heatmap màu (JET)
    heatmap_uint8 = np.uint8(255 * mask)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLOR_BGR2RGB)
    
    # --- FIX QUAN TRỌNG: ĐẢO KÊNH MÀU (BGR -> RGB) ---
    # OpenCV mặc định là BGR (Xanh -> Đỏ). Streamlit cần RGB.
    # Lệnh này sẽ đảo ngược lại: Đỏ -> Vùng quan trọng, Xanh -> Vùng nền
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 3. Trộn màu (Weighted Add)
    # 60% Ảnh gốc + 40% Heatmap
    blended = img_np * 0.6 + heatmap_color * 0.4
    
    return np.clip(blended, 0, 255).astype(np.uint8)

# --- TẤN CÔNG PGD ---
def create_attack(model, image, text_features_all, target_idx, epsilon, step_size, device, dummy_prompt, steps=10):
    if epsilon == 0: return image.clone().detach()
    
    x0 = image.clone().detach().float()
    delta = torch.zeros_like(x0, device=device)
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)
    
    text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

    for _ in range(steps):
        x_in = x0 + delta
        img_emb = model.encode_image(pmg_img_preprocessing(x_in), dummy_prompt)
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        logits = img_norm @ text_features_all.T
        loss = F.cross_entropy(logits, target_idx)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = delta.grad.detach()
            delta.data = torch.clamp(delta + step_size * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, -x0, 1.0 - x0)
            delta.grad.zero_()
            
    return (x0 + delta).detach()

# --- DỰ ĐOÁN ZERO-SHOT ---
def predict_zero_shot(model, image_tensor, text_features, dummy_prompt):
    with torch.no_grad():
        img_emb = model.encode_image(pmg_img_preprocessing(image_tensor.float()), dummy_prompt)
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        logits = (100.0 * img_norm @ text_features.T).softmax(dim=-1)
        conf, idx = logits[0].max(0)
    return conf, idx

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models(model_name, ckpt_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.sidebar.text(f"⏳ Đang khởi tạo: {model_name}...")
    
    model_orig, preprocess = clip.load(model_name, device=device, jit=False)
    model_orig = model_orig.float().eval()
    
    model_robust, _ = clip.load(model_name, device=device, jit=False)
    model_robust = model_robust.float().eval()
    
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            st.sidebar.info(f"📂 Đang đọc file: {os.path.basename(ckpt_path)}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            def find_state_dict(obj, depth=0):
                if depth > 2: return None
                if isinstance(obj, dict):
                    keys = list(obj.keys())
                    if any("weight" in k for k in keys[:20]): 
                        return obj
                    for v in obj.values():
                        res = find_state_dict(v, depth+1)
                        if res: return res
                return None

            state_dict = find_state_dict(checkpoint)
            if state_dict is None: state_dict = checkpoint

            new_state_dict = OrderedDict()
            keys = list(state_dict.keys())
            
            is_visual_only = False
            if "conv1.weight" in keys and "visual.conv1.weight" not in keys:
                is_visual_only = True
                st.sidebar.warning("⚠️ Visual-Only Checkpoint detected -> Auto-fixing prefix.")

            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("module."): new_key = new_key[7:]
                if new_key.startswith("model."): new_key = new_key[6:]
                if is_visual_only: new_key = f"visual.{new_key}"
                new_state_dict[new_key] = v

            model_robust.load_state_dict(new_state_dict, strict=False)
            
            w_orig = model_orig.visual.conv1.weight.flatten()
            w_rob = model_robust.visual.conv1.weight.flatten()
            diff = torch.sum((w_orig - w_rob) ** 2).item()
            
            if diff > 0.001:
                st.sidebar.success(f"✅ Nạp thành công! (Diff: {diff:.2f})")
            else:
                st.sidebar.error("❌ Thất bại: Trọng số không đổi.")
                
        except Exception as e:
            st.sidebar.error(f"Lỗi nạp file: {e}")
            
    return model_orig, model_robust, preprocess, device

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    model_name = st.selectbox(
        "Kiến trúc Model", 
        ["ViT-B/32", "RN50", "RN101", "ViT-B/16", "ViT-L/14"],
        index=0 
    )
    
    ckpt_path = st.text_input("Đường dẫn file .pth/.tar", "pt_models/cosine_loss_cifar100_20251221_064917.pth.tar")
    
    st.divider()
    st.subheader("Cấu hình Tấn công PGD")
    st.caption("Nhập giá trị Pixel (0-255)")
    
    eps_pixel = st.number_input("1. Epsilon (Pixel)", min_value=0, max_value=32, value=6)
    steps = st.number_input("2. Steps (Số bước)", min_value=1, max_value=50, value=6)
    alpha_pixel = st.number_input("3. Step Size (Pixel)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    epsilon = eps_pixel / 255.0
    step_size = alpha_pixel / 255.0
    
    st.info(f"Eps={eps_pixel}/255, Steps={steps}, Alpha={alpha_pixel}/255")

# Load Models
model_orig, model_robust, preprocess, device = load_all_models(model_name, ckpt_path)

col_up1, col_up2 = st.columns([1, 2])
with col_up1:
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "png", "jpeg"])
with col_up2:
    labels_input = st.text_input("Nhập nhãn (ngăn cách bằng phẩy)", "dog, cat, car, airplane, bird, monkey, ship, truck")

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    labels = [l.strip() for l in labels_input.split(',')]
    
    if len(labels) < 2:
        st.error("Vui lòng nhập ít nhất 2 nhãn.")
    else:
        text_tokens = clip.tokenize(labels).to(device)
        dummy_prompt = torch.tensor([0]).to(device)

        with torch.no_grad():
            text_f_orig = model_orig.encode_text(text_tokens).float()
            text_f_orig /= text_f_orig.norm(dim=-1, keepdim=True)
            
            text_f_robust = model_robust.encode_text(text_tokens).float()
            text_f_robust /= text_f_robust.norm(dim=-1, keepdim=True)
            
            tfms = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            img_clean = tfms(img_pil).unsqueeze(0).to(device)

        # 1. Dự đoán Clean
        c_conf, c_idx = predict_zero_shot(model_orig, img_clean, text_f_orig, dummy_prompt)
        clean_label = labels[c_idx.item()]

        # 2. Tấn công & GradCAM
        with torch.enable_grad():
            img_adv = create_attack(
                model_orig, img_clean, text_f_orig, c_idx.view(1), 
                epsilon=epsilon, 
                step_size=step_size,
                device=device, 
                dummy_prompt=dummy_prompt, 
                steps=steps
            )
            
            # Chọn Layer quan trọng (LN_1 cho ViT)
            if "RN" in model_name: 
                target_layer_orig = model_orig.visual.layer4
                target_layer_rob = model_robust.visual.layer4
            else: 
                target_layer_orig = model_orig.visual.transformer.resblocks[-1].ln_1
                target_layer_rob = model_robust.visual.transformer.resblocks[-1].ln_1

            cam_orig = ClipGradCAM(model_orig, target_layer_orig)
            cam_rob = ClipGradCAM(model_robust, target_layer_rob)
            
            mask_clean = cam_orig(img_clean.clone().requires_grad_(True), text_f_orig[c_idx : c_idx+1], device, dummy_prompt)
            mask_adv_orig = cam_orig(img_adv.clone().requires_grad_(True), text_f_orig[c_idx : c_idx+1], device, dummy_prompt)
            mask_adv_rob = cam_rob(img_adv.clone().requires_grad_(True), text_f_robust[c_idx : c_idx+1], device, dummy_prompt)

        # --- HIỂN THỊ ---
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.image(img_pil.resize((224,224)), caption="Ảnh Gốc")
        col2.metric("CLIP Gốc", clean_label, f"{c_conf.item()*100:.1f}%")
        col3.image(overlay_heatmap(img_pil, mask_clean), caption="Vùng chú ý (Clean)")

        st.subheader(f"☠️ Kết quả PGD (Eps={eps_pixel}/255, Steps={steps})")
        
        adv_np = img_adv.squeeze().cpu().numpy().transpose(1, 2, 0)
        adv_pil = Image.fromarray((np.clip(adv_np * 255, 0, 255)).astype(np.uint8))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ❌ CLIP Gốc")
            conf, idx = predict_zero_shot(model_orig, img_adv, text_f_orig, dummy_prompt)
            st.image(overlay_heatmap(adv_pil, mask_adv_orig), use_container_width=True, caption="Heatmap Gốc")
            
            is_correct = (idx == c_idx)
            st.metric("Dự đoán", labels[idx], delta="Đúng" if is_correct else "Bị lừa", delta_color="normal" if is_correct else "inverse")
            st.caption(f"Độ tin cậy: {conf.item()*100:.2f}%")
            
        with c2:
            st.markdown("### ✅ CLIP Robust")
            conf_r, idx_r = predict_zero_shot(model_robust, img_adv, text_f_robust, dummy_prompt)
            st.image(overlay_heatmap(adv_pil, mask_adv_rob), use_container_width=True, caption="Heatmap Robust")
            
            is_correct_r = (idx_r == c_idx)
            st.metric("Dự đoán", labels[idx_r], delta="Vững vàng" if is_correct_r else "Thất bại", delta_color="normal" if is_correct_r else "inverse")
            st.caption(f"Độ tin cậy: {conf_r.item()*100:.2f}%")

        buf = io.BytesIO()
        adv_pil.save(buf, format="PNG")
        st.download_button("📥 Tải ảnh Adversarial", buf.getvalue(), f"adv_clip_eps_{eps_pixel}.png", "image/png")