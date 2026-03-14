import numpy as np
import cv2
import os
from datetime import datetime

def evaluate_disparity_no_gt(disp: np.ndarray, 
                            model_name: str = "unknown_model",
                            max_disp: float = 240.0,         
                            export_txt: bool = True,
                            export_csv: bool = True) -> dict:
    """
    No-Reference Disparity Evaluation

    Input:

    disp: (H, W) np.float32 Left disparity map (divided by 16)
    model_name: Model name, used to generate the report file name
    max_disp: Maximum reasonable disparity (values ​​greater than this are considered invalid/infinity)
    export_txt: Whether to export a readable .txt report
    export_csv: Whether to export a .csv report (for easy comparison of multiple models)

    Return:
        dict contains all evaluation metrics
        results = {
            "Model": model_name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Valid_Pixels_%": round(valid_ratio, 3),
            "Mean_Disparity": round(mean_disp, 3),
            "Smoothness (grad)" : round(smoothness, 4),
            "Edge_Correlation" : round(edge_corr, 4),
            "Speckle_Ratio_%" : round(speckle_ratio, 3),
            "Histogram_Entropy" : round(entropy, 4),
        }
 
    """
    assert disp.ndim == 2, "disparity map must be 2D (H, W)"
    disp = disp.astype(np.float32)
    H, W = disp.shape

    # 1. Effective pixel ratio (>0 and <= max_disp)
    valid_mask = (disp > 0) & (disp <= max_disp)
    valid_ratio = valid_mask.mean() * 100.0

    # 2. Parallax gradient smoothness (the smaller the value, the smoother the surface and the more natural the scene).
    grad_y, grad_x = np.gradient(disp)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    smoothness = grad_mag[valid_mask].mean() if valid_mask.any() else 999.9

    # 3. Edge consistency index (parallax edges should align with intensity edges)
    # We can simply correlate the x-direction gradient and intensity gradient of the left disparity map (using disp itself as a proxy).
    disp_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    intensity_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)  # If the original image is unavailable, use disparity as an intensity approximation.
    edge_corr = np.corrcoef(disp_x[valid_mask].ravel(), intensity_x[valid_mask].ravel())[0, 1]
    edge_corr = 0.0 if np.isnan(edge_corr) else edge_corr

    # 4. Noise ratio (isolated small spots)
    # Binarize the disparity > 0, then perform connected component analysis.
    binary = (disp > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    component_sizes = np.bincount(labels.ravel())[1:]  
    speckle_ratio = (component_sizes < 50).sum() / len(component_sizes) * 100.0 if len(component_sizes) > 0 else 0.0

    # 5.Parallax histogram entropy (the more uniform the entropy, the more random and lower the quality of the match).
    hist, _ = np.histogram(disp[valid_mask], bins=100, range=(0.1, max_disp), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-12))

    # 6. Average parallax (can be used as a reference for scene distance)
    mean_disp = disp[valid_mask].mean() if valid_mask.any() else 0.0

    # Summarize the results
    results = {
        "Model"              : model_name,
        "Timestamp"          : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Valid_Pixels_%"     : round(valid_ratio, 3),
        "Mean_Disparity"     : round(mean_disp, 3),
        "Smoothness (grad)"  : round(smoothness, 4),
        "Edge_Correlation"   : round(edge_corr, 4),
        "Speckle_Ratio_%"    : round(speckle_ratio, 3),
        "Histogram_Entropy"  : round(entropy, 4),
    }

    # Export Report
    os.makedirs("disparity_reports", exist_ok=True)
    base_name = f"disparity_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if export_txt:
        txt_path = f"disparity_reports/{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("===  Disparity map without true value quality assessment report ===\n\n")
            f.write(f"Model Name        : {model_name}\n")
            f.write(f"Assessment time   : {results['Timestamp']}\n")
            f.write(f"Resolution        : {W} x {H}\n")
            f.write(f"Maximum disaprity : {max_disp}\n\n")
            f.write("Indicator Explanation and Values : \n")
            f.write("-" * 40 + "\n")
            f.write(f"Effective pixel ratio         : {results['Valid_Pixels_%']}%   (The higher the better)\n")
            f.write(f"Mean disaprity                : {results['Mean_Disparity']} px\n")
            f.write(f"Smoothness (mean gradient)    : {results['Smoothness (grad)']}   (The smaller the better)\n")
            f.write(f"Marginal consistency correlation coefficient    : {results['Edge_Correlation']}   (The closer to 1, the better.)\n")
            f.write(f"Noise ratio (<50px)      : {results['Speckle_Ratio_%']}%   (The smaller the better)\n")
            f.write(f"Disparity histogram entropy          : {results['Histogram_Entropy']}   (The smaller the value, the more concentrated and better the distribution.)\n")
            f.write("\nConclusions and Recommendations : \n")
            if valid_ratio < 70:
                f.write("  → The effective area is too small. It is recommended to increase numDisparities or decrease uniquenessRatio.\n")
            if smoothness > 2.0:
                f.write("  → The disparity map is too coarse; it is recommended to increase P1/P2 or perform post-processing smoothing.\n")
            if speckle_ratio > 15:
                f.write("  → Too much noise, it is recommended to increase speckleWindowSize/speckleRange.\n")
        print(f"The report has been stored : {txt_path}")

    if export_csv:
        csv_path = f"disparity_reports/disparity_summary.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(",".join(results.keys()) + "\n")
            f.write(",".join([str(v) for v in results.values()]) + "\n")
        print(f"Abstract has been added to : {csv_path}")

    return results