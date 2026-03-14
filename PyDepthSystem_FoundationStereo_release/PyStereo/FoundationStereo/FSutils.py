import sys
import os
import torch
import cv2
import numpy as np


def load_ckpt(model, ckpt_path, strict=True):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 常見情況：checkpoint 可能是 {'model': state_dict} 或直接 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 去除可能的 'module.' 前綴（DataParallel 儲存時會加）
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 載入（strict=False 允許部分 key 不匹配，debug 時方便）
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing:
        print(f"Missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]} ...")
    
    print(f"Loaded checkpoint from {ckpt_path}")
    return model




def preprocess(left_img, right_img, target_height=None, target_width=None):
    """
    輸入：BGR uint8 numpy array (H, W, 3)
    輸出：兩個 torch.Tensor [3, H', W']，已 normalize 到 [-1,1]
    """

    # BGR → RGB
    left_rgb  = cv2.cvtColor(left_img,  cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # uint8 → float32 [0,1]
    left_rgb  = left_rgb.astype(np.float32) / 255.0
    right_rgb = right_rgb.astype(np.float32) / 255.0

    # 轉 tensor [3, H, W]
    left_tensor  = torch.from_numpy(left_rgb.transpose(2, 0, 1)).float()
    right_tensor = torch.from_numpy(right_rgb.transpose(2, 0, 1)).float()

    # Normalize to [-1, 1] （ImageNet 風格，MobileNet 常用）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    left_tensor  = (left_tensor  - mean) / std
    right_tensor = (right_tensor - mean) / std

    return left_tensor, right_tensor