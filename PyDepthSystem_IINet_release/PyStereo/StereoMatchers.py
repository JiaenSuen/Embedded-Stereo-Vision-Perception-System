import sys
import os
import torch
import numpy as np
import cv2
from torchvision import transforms

 
# Assume IINet is added as a submodule
from .IINet.options import Options
from .IINet.modules.disp_model import DispModel

from pathlib import Path
THIS_FILE_DIR = Path(__file__).resolve().parent

class IINetv1:
    max_disp= 256  # 官方 max_disp: 192

    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_model(ckpt_path=None):
        if ckpt_path is None:
            raise ValueError("")

        # dot_model.yaml 的設定，讓 DispModel 架構與 checkpoint 匹配
        opts = Options()
        opts.image_encoder_name = "efficientnet"
        opts.matching_encoder_type = "unet"
        opts.feature_volume_type = "ms_cost_volume"
        opts.dot_dim = 1
        opts.cv_encoder_type = "multi_scale_encoder"
        opts.depth_decoder_name = "unet_pp"
        opts.matching_scale = 2
        opts.multiscale = 2
        opts.out_scale = 4
        opts.max_disp = 192


        model = DispModel(opts)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict, strict=True)  # 先用 True，若仍有 mismatch 改 False
        model.to(IINetv1.device)
        model.eval()
        return model

    @staticmethod
    def stereo_matching(left_img, right_img ):
        if IINetv1.model is None:
            ckpt_path = THIS_FILE_DIR / "IINet" / "pretrained_models" / "sceneflow.tar" 
            # 若已解壓，試 "path/to/data.pkl"（確保 data/ 資料夾在同目錄）
            IINetv1.model = IINetv1._load_model(ckpt_path)

        # 記錄原始大小（來自 rectified 影像）
        original_h, original_w = left_img.shape[:2]

        # 轉 RGB
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # 新增：Resize 輸入圖片以加速（可調整 target_height）
        target_height = 512  # 可改為 512、256 等，越大精度越高但越慢
        scale_factor = target_height / original_h  # <1 表示 downsample
        target_width = int(original_w * scale_factor)

        left_resized = cv2.resize(left_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(right_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # 更新 h, w 為 resize 後大小
        h, w = target_height, target_width
        left_rgb = left_resized
        right_rgb = right_resized
        # Pad 到 32 倍數（top & right，官方一致）
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        left_padded = np.pad(left_rgb, ((pad_h, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        right_padded = np.pad(right_rgb, ((pad_h, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        # ImageNet normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        left_tensor = transform(left_padded).unsqueeze(0).to(IINetv1.device)
        right_tensor = transform(right_padded).unsqueeze(0).to(IINetv1.device)

        inputs = {'left': left_tensor, 'right': right_tensor}
        with torch.no_grad():
            outputs = IINetv1.model(inputs)
        disp_pred = outputs['disp_pred_s0']
        disp_scaled = 16.0 * disp_pred
        disp_np = disp_scaled.squeeze().cpu().numpy()
        # Crop 回 resize 後的大小（移除 top pad_h & right pad_w）
        disp = disp_np[pad_h : pad_h + h, :w]  # 正確 crop：:w 移除 right pad
        # Upscale 回原始大小，並 scale 視差值（因為 downsample 時視差值需放大）
        upscale_factor = 1.0 / scale_factor  # >1
        disp_up = cv2.resize(disp, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        disp_scaled_back = disp_up * upscale_factor
        # 最終調整到指定大小 1600x1200（寬1600、高1200）
        desired_w, desired_h = 1600, 1200
        disp_final = cv2.resize(disp_scaled_back, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
 
        return disp_final.astype(np.float32)
