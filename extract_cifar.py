import pickle
import numpy as np
from PIL import Image
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar100_to_png(file_path, output_dir, num_images=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nạp dữ liệu
    data_dict = unpickle(file_path)
    images = data_dict[b'data']
    filenames = data_dict[b'filenames']
    
    # CIFAR-100 có cấu trúc: 10000 ảnh, mỗi ảnh 3072 pixels (32x32x3)
    # Dữ liệu được sắp xếp theo Red, Green, Blue
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    print(f"Đang trích xuất {num_images} ảnh từ {file_path}...")
    
    for i in range(min(num_images, len(images))):
        img_array = images[i]
        img_name = filenames[i].decode('utf-8')
        
        img = Image.fromarray(img_array)
        # Lưu ảnh vào thư mục
        img.save(os.path.join(output_dir, img_name))
        
    print(f"✅ Hoàn thành! Ảnh được lưu tại: {output_dir}")

# Thực hiện trích xuất
save_cifar100_to_png('./cifar100/test', 'extracted_images', num_images=1000)