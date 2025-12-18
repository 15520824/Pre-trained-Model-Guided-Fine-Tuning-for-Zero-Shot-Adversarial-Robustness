import streamlit as st
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import torch.nn.functional as F

# --- CẤU HÌNH ---
st.set_page_config(page_title="Zero-Shot Robustness Demo", layout="wide")

# --- CLASS GRADCAM TỰ VIẾT ---
class ClipGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Đăng ký hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image_input, text_features, device):
        # 1. Forward Image
        dummy_prompt = torch.tensor([0]).to(device)
        image_features = self.model.encode_image(image_input, dummy_prompt)
        
        # 2. Tính Score
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features_norm * text_features_norm).sum()
        
        # 3. Backward
        self.model.zero_grad()
        score.backward()
        
        # 4. Tính CAM
        gradients = self.gradients
        activations = self.activations
        
        # Kiểm tra nếu không bắt được gradient (tránh lỗi None)
        if gradients is None or activations is None:
            return np.zeros((7, 7), dtype=np.float32)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        return cam

# --- HÀM XỬ LÝ ẢNH HEATMAP (ĐÃ FIX LỖI OPENCV) ---
def overlay_heatmap(img_pil, cam_mask):
    # 1. Ép kiểu sang float32 ngay lập tức để tránh lỗi OpenCV func != 0
    cam_mask = cam_mask.astype(np.float32)

    # 2. Chuẩn hóa mask về 0-1
    cam_mask = cam_mask - np.min(cam_mask)
    cam_mask = cam_mask / (np.max(cam_mask) + 1e-8)
    
    # 3. Chuyển ảnh gốc sang Numpy
    img = np.array(img_pil)
    h, w = img.shape[:2]
    
    # 4. Resize mask (Giờ đã an toàn vì là float32)
    heatmap = cv2.resize(cam_mask, (w, h))
    
    # 5. Tô màu
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 6. Trộn ảnh
    # Đảm bảo ảnh gốc và heatmap cùng kích thước và kiểu dữ liệu
    if len(img.shape) == 2: # Nếu là ảnh đen trắng
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    result = heatmap * 0.4 + img * 0.6
    return np.uint8(result)

# --- LOAD MODEL ---
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Dùng RN50 vì ViT không có layer4 để soi GradCAM
    model, preprocess = clip.load("RN50", device=device) 
    model.eval()
    return model, preprocess, device

with st.spinner("Đang khởi động hệ thống..."):
    model, preprocess, device = load_clip_model()

# --- GIAO DIỆN ---
st.title("🛡️ Demo: Zero-Shot Adversarial Robustness")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Input")
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "png", "jpeg"])
    labels_input = st.text_area("Nhãn (cách nhau dấu phẩy)", value="a dog, a cat, a car, a plane")
    show_heatmap = st.checkbox("Hiển thị Heatmap (Grad-CAM)", value=True)
    btn = st.button("Phân tích", type="primary")

with col2:
    st.header("2. Kết quả")
    if uploaded_file and btn:
        img_pil = Image.open(uploaded_file).convert("RGB")
        
        # --- CLASSIFICATION ---
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        labels = [l.strip() for l in labels_input.split(',')]
        text = clip.tokenize(labels).to(device)
        
        with torch.no_grad():
            dummy = torch.tensor([0]).to(device)
            img_emb = model.encode_image(img_tensor, dummy)
            text_emb = model.encode_text(text)
            
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_emb @ text_emb.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(labels))
            
        scores = values.cpu().numpy() * 100
        top_labels = [labels[idx] for idx in indices.cpu().numpy()]
        st.bar_chart(pd.DataFrame({"Label": top_labels, "Score": scores}).set_index("Label"))
        
        top_idx = indices[0].item()
        st.success(f"Dự đoán: **{top_labels[0]}** ({scores[0]:.2f}%)")

        # --- HEATMAP ---
        if show_heatmap:
            with st.spinner("Đang vẽ Heatmap..."):
                try:
                    with torch.enable_grad():
                        target_layer = model.visual.layer4
                        grad_cam = ClipGradCAM(model, target_layer)
                        
                        target_text_emb = text_emb[top_idx].unsqueeze(0)
                        
                        # Chạy GradCAM
                        mask = grad_cam(img_tensor, target_text_emb, device)
                        
                        # Vẽ ảnh
                        heatmap_img = overlay_heatmap(img_pil, mask)
                        
                        st.image(heatmap_img, caption=f"AI đang nhìn vào đâu để nhận ra '{top_labels[0]}'?", use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi Heatmap: {e}")
                    import traceback
                    st.text(traceback.format_exc()) # In chi tiết lỗi nếu còn

    elif uploaded_file:
        st.image(uploaded_file, caption="Ảnh gốc", width=400)