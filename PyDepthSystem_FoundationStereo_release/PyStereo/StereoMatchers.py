import torch
import cv2
import numpy as np
from .FoundationStereo.core.foundation_stereo import FoundationStereo
from .FoundationStereo.FSutils import preprocess  
import argparse 
from pathlib import Path
import math


BASE_DIR = Path(__file__).resolve().parent
class FoundationStereoV1:
    max_disp = 256 #192

    @staticmethod
    def pad_to_32_multiple(img):
        """Pad image so that H and W are multiples of 32"""
        h, w = img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h == 0 and pad_w == 0:
            return img, (0, 0, 0, 0)  # top, bottom, left, right
        # Use reflect to avoid boundary artifacts (better than constant or replicate).
        padded = cv2.copyMakeBorder(
            img,
            top=0, bottom=pad_h,
            left=0, right=pad_w,
            borderType=cv2.BORDER_REFLECT_101
        )
        return padded, (0, pad_h, 0, pad_w)
    
    @staticmethod
    def resize_for_foundation(img, target_short=256):
        h, w = img.shape[:2]
        short = min(h, w)
        if short <= target_short:
            return img, 1.0


        scale = target_short / short
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale




    def __init__(self, ckpt_path = BASE_DIR / 'FoundationStereo' / 'checkpoints' / 'Vit-small.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128, 128])
        parser.add_argument('--vit_size', type=str, default='vits')
        parser.add_argument('--n_downsample', type=int, default=2)
        parser.add_argument('--mixed_precision', action='store_true', default=True)
        parser.add_argument('--max_disp', type=int, default=self.max_disp)
        parser.add_argument('--valid_iters', type=int, default=32)#32)
        parser.add_argument('--corr_levels', type=int, default=2)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--corr_implementation', type=str, default='reg')
        parser.add_argument('--n_gru_layers', type=int, default=3)

        args = parser.parse_args([])
        args.vit_size = 'vits'
        args.hidden_dims = [128, 128, 128]
        args.max_disp = self.max_disp  
        cfg = vars(args)  # Namespace → dict
        self.model = FoundationStereo(args)
        self.model.to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")
        self.model.eval()
        print(f"FoundationStereo loaded from {ckpt_path} on {self.device}")



    @torch.no_grad()
    def stereo_matching(self, left_img, right_img, target_size=None):
        orig_h, orig_w = left_img.shape[:2]
        left_img, scale = FoundationStereoV1.resize_for_foundation(left_img, 256)
        right_img, _    = FoundationStereoV1.resize_for_foundation(right_img, 256)
        resized_h, resized_w = left_img.shape[:2]
        # Pad to multiple of 32
        left_padded, pad_info = FoundationStereoV1.pad_to_32_multiple(left_img)
        right_padded = cv2.copyMakeBorder(
            right_img,
            top=pad_info[0], bottom=pad_info[1],
            left=pad_info[2], right=pad_info[3],
            borderType=cv2.BORDER_REFLECT_101
        )
        # Preprocessing (using padded graphs)
        left_tensor, right_tensor = preprocess(left_padded, right_padded)

        if left_tensor.dim() == 3:
            left_tensor = left_tensor.unsqueeze(0)
            right_tensor = right_tensor.unsqueeze(0)
        left_tensor = left_tensor.to(self.device)
        right_tensor = right_tensor.to(self.device)

        # inference
        output = self.model(left_tensor, right_tensor)

        # Force extraction of the final disparity tensor (handling possible nested lists/tuples)
        disp_tensor = output
        while isinstance(disp_tensor, (list, tuple)):
            if len(disp_tensor) == 0:
                raise ValueError("The model output is empty. list")
            disp_tensor = disp_tensor[-1]  # Keep taking the last item until it is no longer a list/tuple.

        # Now, disp_tensor should be either tensor [B, C, H, W] or [B, H, W].
        if disp_tensor.dim() == 4:  # [B, C, H, W] , C=1
            disp_tensor = disp_tensor[:, 0, :, :]  # Take channel 0
        elif disp_tensor.dim() == 3:  # [B, H, W]
            disp_tensor = disp_tensor[0, :, :]     # Take batch 0
        disp = disp_tensor.cpu().numpy()


        # Remove all axes with a dimension of 1, and ensure that the final result is 2D.
        disp = np.squeeze(disp)             
        if disp.ndim == 3:
            # If it's still 3D, it might leave (C,H,W) or (H,W,C) or (1,H,W).
            if disp.shape[0] == 1:
                disp = disp[0]                      # (1,H,W) → (H,W)
            elif disp.shape[-1] == 1:
                disp = disp[..., 0]                 # (H,W,1) → (H,W)
            else:
                raise ValueError(f"Unable to process disparity shape: {disp.shape} (after squeeze still 3D)")

        if disp.ndim != 2:
            raise ValueError(f"Disparity not 2D Array, Current shape: {disp.shape}")

        pad_top, pad_bottom, pad_left, pad_right = pad_info
        disp = disp[
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w
        ]
         
        # After confirming the crop size is correct, please note that the dimensions are correct.
        if disp.shape == (resized_h, resized_w):
            print(
                f"Warning: Cropped size mismatch, "
                f"expect ({resized_h}, {resized_w}), actual {disp.shape}"
            )

        # Post-processing
        disp = disp.astype(np.float32)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)

        if scale != 1.0:
            disp = cv2.resize(
                disp,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            disp = disp / scale
        disp[disp < 0] = 0
        disp[disp > self.max_disp] = self.max_disp
        disp = cv2.medianBlur(disp, 5)
        return disp
    

