import cv2
import numpy as np
import os
from PyStereo.StereoLib import *
from PyStereo.StereoMatchers import *
from PyStereo.DetectorLib import *
import shutil

 

def evaluate_sgbm_models(dataset_path, model, model_name="", class_names=["Box", "Bottle", "Doll","Battery"], max_range_cm=300):
    # Load stereo rectification parameters
    stereo_rectify_params = load_stereo_camera_params(npz_file="stereo_rectify_params.npz")
    
    # Collect image filenames from 'left' folder (assuming PNG files)
    left_dir   = os.path.join(dataset_path, "stereoLeft")
    right_dir  = os.path.join(dataset_path, "stereoRight")
    labels_dir = os.path.join(dataset_path, "stereoLabel")
    
    image_files = [f for f in os.listdir(left_dir) if f.endswith(".png")]
    if not image_files:
        raise ValueError("No PNG images found in the 'left' folder.")
    
    # Dictionary to store errors for the model: model_name -> list of (estimated_depth, gt_depth) pairs
    model_errors = {model_name: []}
    
    for img_file in image_files:
        # Load left and right images
        left_path = os.path.join(left_dir, img_file)
        right_path = os.path.join(right_dir, img_file.replace("left_", "right_"))  # Assuming naming convention like left_xxx.png -> right_xxx.png
        label_path = os.path.join(labels_dir, img_file.replace(".png", ".txt"))  # labels with same name as left but .txt
        
        if not os.path.exists(right_path):
            print(f"Right image not found: {right_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        raw = left_img  # Use left as reference
        
        # Rectify images
        left_rect, right_rect = rectify_stereo_images(left_img, right_img, rectify_params=stereo_rectify_params, show=False)
        
        # Load YOLO labels with extra distance column
        labels = []
        img_h, img_w = raw.shape[:2]
        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 6:  # class_id cx cy w h distance_cm
                        continue
                    
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    gt_distance_cm = float(parts[5])  # Ground truth distance
                    
                    class_name = class_names[class_id] if class_names else str(class_id)
                    
                    # Convert to pixel coordinates
                    cx_p = cx * img_w
                    cy_p = cy * img_h
                    w_p = w * img_w
                    h_p = h * img_h
                    
                    labels.append((class_name, cx_p, cy_p, w_p, h_p, gt_distance_cm))
        except FileNotFoundError:
            print(f"Label file not found: {label_path}")
            continue
        
        if not labels:
            print(f"No valid labels in: {label_path}")
            continue
        
        # Compute disparity using the provided model function
        disp = model(left_rect, right_rect)  # Assuming model is a callable function
        
        # Compute depth map
        depth_map_mm = cv2.reprojectImageTo3D(disp, stereo_rectify_params["Q"])[:,:,2]
        depth_map_mm[disp <= 0] = 0
        depth_map_mm_clipped = np.clip(depth_map_mm, 0, max_range_cm * 10)
        depth_map_cm = depth_map_mm_clipped / 10.0
        
        # For each label, estimate depth at center and collect error
        for _, cx_p, cy_p, _, _, gt_distance_cm in labels:
            x_center = int(cx_p)
            y_center = int(cy_p)
            if 0 <= x_center < img_w and 0 <= y_center < img_h:
                est_depth_cm = depth_map_cm[y_center, x_center]
                if est_depth_cm > 0:  # Valid estimate
                    model_errors[model_name].append((est_depth_cm, gt_distance_cm))
    
    # Compute average absolute errors
    report_lines = ["SGBM Models Evaluation Report\n"]
    for m_name, errors in model_errors.items():
        if not errors:
            report_lines.append(f"{m_name}: No valid detections for evaluation.")
            continue
        
        abs_errors = [abs(est - gt) for est, gt in errors]
        mean_abs_error = np.mean(abs_errors)
        num_detections = len(errors)
        
        report_lines.append(f"{m_name}:")
        report_lines.append(f"  Number of detections: {num_detections}")
        report_lines.append(f"  Mean Absolute Error (cm): {mean_abs_error:.2f}\n")
    
    # Dynamically generate output path based on model_name
    if not model_name:
        model_name = "default"  # Fallback if no name provided
    output_report_path = f"output/{model_name}_Result.txt"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Write report to TXT file
    with open(output_report_path, "w") as f:
        f.writelines("\n".join(report_lines))
    
    print(f"Evaluation report saved to: {output_report_path}")

if __name__ == "__main__":

    def use_FS(): 
        fs_stereo = FoundationStereoV1(device='cuda')
        return fs_stereo.stereo_matching, "FoundationStereoV1" 
     
    model , model_name = use_FS()
    evaluate_sgbm_models(dataset_path='input/OBJ5', model=model , model_name=model_name   , max_range_cm=200)