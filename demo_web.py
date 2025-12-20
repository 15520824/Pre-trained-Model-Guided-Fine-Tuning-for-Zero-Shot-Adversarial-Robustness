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

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="🛡️ Zero-Shot Robustness Lab", layout="wide")

# --- CLASS GRADCAM ---
class ClipGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, image_input, text_features, device, dummy_prompt):
        self.model.zero_grad()
        img_emb = self.model.encode_image(image_input.float(), dummy_prompt)
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        score = (img_norm * text_features).sum()
        score.backward(retain_graph=True)
        
        if self.gradients is None: return np.zeros((7, 7))
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))
        return cam.squeeze().cpu().detach().numpy()

# --- HÀM TẠO NHIỄU FGSM ---
def create_attack(model, image, text_features, epsilon, device, dummy_prompt):
    if epsilon == 0: return image.clone().detach()
    
    img_adv = image.clone().detach().requires_grad_(True).float()
    img_emb = model.encode_image(img_adv, dummy_prompt)
    img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    loss = (img_norm * text_features).sum()
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        img_adv = img_adv + epsilon * img_adv.grad.sign()
        img_adv = torch.clamp(img_adv, -1, 1)
    return img_adv.detach()

# --- DỰ ĐOÁN ZERO-SHOT ---
def predict_zero_shot(model, image_tensor, text_features, dummy_prompt):
    with torch.no_grad():
        img_emb = model.encode_image(image_tensor.float(), dummy_prompt)
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        logits = (100.0 * img_norm @ text_features.T).softmax(dim=-1)
        conf, idx = logits[0].max(0)
    return conf, idx

# --- HÀM TRỘN HEATMAP ---
def overlay_heatmap(img_pil, mask):
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    img_resized = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(mask, (224, 224))), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Chuyển BGR của OpenCV sang RGB
    return np.uint8(heatmap * 0.4 + img_resized * 0.6)

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models(ckpt_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_orig, preprocess = clip.load("RN50", device=device, jit=False)
    model_orig = model_orig.float().eval()
    
    model_robust, _ = clip.load("RN50", device=device, jit=False)
    model_robust = model_robust.float().eval()
    
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            model_robust.load_state_dict(new_state_dict, strict=False)
            st.sidebar.success("✅ Đã nạp checkpoint thành công!")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Lỗi nạp file: {e}")
    return model_orig, model_robust, preprocess, device

# --- GIAO DIỆN ---
st.title("🛡️ Zero-Shot Adversarial Robustness Lab")

with st.sidebar:
    st.header("Cấu hình")
    ckpt_path = st.text_input("Đường dẫn file .pth/.tar", "checkpoint.pth.tar")
    epsilon = st.slider("Cường độ nhiễu (Epsilon)", 0.0, 0.1, 0.0, step=0.01)

model_orig, model_robust, preprocess, device = load_all_models(ckpt_path)

col_up1, col_up2 = st.columns([1, 2])
with col_up1:
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "png", "jpeg"])
with col_up2:
    labels_input = st.text_input("Nhãn phân loại", value="dog, cat, car, tree")

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    labels = [l.strip() for l in labels_input.split(',')]
    text_tokens = clip.tokenize(labels).to(device)
    dummy_prompt = torch.tensor([0]).to(device)

    # Trích xuất & Chuẩn hóa text features
    with torch.no_grad():
        text_f_orig = model_orig.encode_text(text_tokens).float()
        text_f_orig /= text_f_orig.norm(dim=-1, keepdim=True)
        
        text_f_robust = model_robust.encode_text(text_tokens).float()
        text_f_robust /= text_f_robust.norm(dim=-1, keepdim=True)

        img_clean_tensor = preprocess(img_pil).unsqueeze(0).to(device).float()

    # 1. Dự đoán ảnh sạch
    c_conf, c_idx = predict_zero_shot(model_orig, img_clean_tensor, text_f_orig, dummy_prompt)
    clean_label = labels[c_idx.item()]

    # 2. Tạo ảnh nhiễu & Grad-CAM
    with torch.enable_grad():
        img_adv_tensor = create_attack(model_orig, img_clean_tensor, text_f_orig[c_idx : c_idx+1], epsilon, device, dummy_prompt)
        
        cam_orig_obj = ClipGradCAM(model_orig, model_orig.visual.layer4)
        cam_robust_obj = ClipGradCAM(model_robust, model_robust.visual.layer4)

        mask_clean = cam_orig_obj(img_clean_tensor.clone().requires_grad_(True), text_f_orig[c_idx : c_idx+1], device, dummy_prompt)
        mask_adv_orig = cam_orig_obj(img_adv_tensor.clone().requires_grad_(True), text_f_orig[c_idx : c_idx+1], device, dummy_prompt)
        mask_adv_robust = cam_robust_obj(img_adv_tensor.clone().requires_grad_(True), text_f_robust[c_idx : c_idx+1], device, dummy_prompt)

    # --- HIỂN THỊ KẾT QUẢ ---
    st.header("📊 Phân tích chi tiết")
    
    # Hàng 1: Clean
    st.subheader("1. Phân tích ảnh Gốc (Clean)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_pil.resize((224,224)), caption="Ảnh sạch", width=350)
        st.metric("Dự đoán gốc", clean_label, f"{c_conf.item()*100:.2f}%")
    with col2:
        st.image(overlay_heatmap(img_pil, mask_clean), caption="Vùng chú ý Clean", width=350)

    st.divider()

    # Hàng 2: Adversarial
    st.subheader(f"2. Đối phó tấn công FGSM (Epsilon = {epsilon})")
    res1, res2 = st.columns(2)

    # Khôi phục ảnh nhiễu từ Tensor sang PIL để hiển thị và download
    # (Ảnh CLIP preprocess thường nằm trong khoảng [-1, 1], cần đưa về [0, 255])
    adv_np = img_adv_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # Denormalize (Lưu ý: Đây là bước xấp xỉ để hiển thị, ảnh gốc trong Tensor vẫn chuẩn)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8)
    adv_pil = Image.fromarray((adv_np * 255).astype(np.uint8))

    with res1:
        st.error("❌ CLIP Nguyên bản")
        conf, idx = predict_zero_shot(model_orig, img_adv_tensor, text_f_orig, dummy_prompt)
        st.image(overlay_heatmap(adv_pil, mask_adv_orig), use_container_width=True)
        delta_status = "Bị tấn công" if idx != c_idx else "Đúng"
        st.metric(label="Dự đoán", value=labels[idx.item()], delta=delta_status, delta_color="inverse")
        st.caption(f"Tin cậy: {conf.item()*100:.2f}%")

    with res2:
        st.success("✅ CLIP Robust")
        conf_r, idx_r = predict_zero_shot(model_robust, img_adv_tensor, text_f_robust, dummy_prompt)
        st.image(overlay_heatmap(adv_pil, mask_adv_robust), use_container_width=True)
        delta_status_r = "Giữ vững" if idx_r == c_idx else "Bị lừa"
        st.metric(label="Dự đoán", value=labels[idx_r.item()], delta=delta_status_r)
        st.caption(f"Tin cậy: {conf_r.item()*100:.2f}%")

    # --- NÚT DOWNLOAD ---
    st.divider()
    buf = io.BytesIO()
    adv_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 Tải ảnh Adversarial (đã thêm nhiễu)",
        data=buf.getvalue(),
        file_name=f"adversarial_eps_{epsilon}.png",
        mime="image/png"
    )