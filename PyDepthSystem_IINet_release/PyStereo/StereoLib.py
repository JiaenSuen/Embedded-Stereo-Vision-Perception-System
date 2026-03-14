import cv2
import numpy as np
import matplotlib.pyplot as plt



def load_stereo_camera_params(npz_file = "stereo_rectify_params.npz"):
    data = np.load(npz_file, allow_pickle=True)
    return {
        "mapLx" : data["mapLx"],
        "mapLy" : data["mapLy"],
        "mapRx" : data["mapRx"],
        "mapRy" : data["mapRy"],
        "Q"     : data["Q"]     
    }


def rectify_stereo_images(left_image,right_image,rectify_params , show=False , save = False):
    left_rect  = cv2.remap(left_image,  rectify_params["mapLx"], rectify_params["mapLy"], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, rectify_params["mapRx"], rectify_params["mapRy"], cv2.INTER_LINEAR)
    if save :
        cv2.imwrite("_outRect_left.png",left_rect)
    if show :
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(left_rect,  cv2.COLOR_BGR2RGB)); plt.title("Left Rectified")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB)); plt.title("Right Rectified")
        plt.show()
    return left_rect,right_rect




def save_depth_plasma(depth_map, save_path, vmin=None, vmax=None):
    """
    depth_map: float32 (meters)
    save_path: xxx.png
    """

    plt.figure(figsize=(8, 6))
    plt.axis("off")

    plt.imshow(
        depth_map,
        cmap="plasma",
        vmin=vmin if vmin is not None else np.percentile(depth_map, 1),
        vmax=vmax if vmax is not None else np.percentile(depth_map, 99)
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()


# Visualize

def visualize_depth_map(left_img,disp,num_disp,rectify_params,model_name="",file_serial_number="",max_range_cm=110,show=False,output_dir="output"):
 
    disp_vis = np.clip((disp) / num_disp, 0, 1)
    depth_map_mm = cv2.reprojectImageTo3D(disp, rectify_params["Q"])[:,:,2]
    depth_map_mm[disp <= 0] = 0       
    depth_map_mm_clipped = depth_map_mm.copy()
    depth_map_mm_clipped[depth_map_mm > max_range_cm*10] = max_range_cm*10
    depth_map_cm = depth_map_mm_clipped / 10.0


    
 
    plt.figure(figsize=(18,10))
    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    plt.title("Left Rectified", fontsize=16)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='Original Image')
    plt.gca().set_aspect('equal') 

    plt.subplot(2,2,2)
    plt.imshow(disp_vis, cmap='plasma')
    plt.title(f"Disparity (0~{num_disp})", fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04, label='disparity (pixels)')
    
 
    plt.subplot(2,2,3)
    im = plt.imshow(depth_map_cm, cmap='jet', vmin=0, vmax=max_range_cm)

 
    depth_map_vis = depth_map_cm.copy()
    depth_map_vis[depth_map_mm == 0] = 0

 
    depth_norm = depth_map_cm / max_range_cm
    depth_vis = plt.cm.jet(depth_norm)[:, :, :3]
    depth_vis[depth_map_mm == 0] = 0
    cv2.imwrite(f"{output_dir}/depthmap_{file_serial_number}_{model_name}.png", (depth_vis*255).astype(np.uint8)[:,:,::-1])
    
    plt.title(f"Depth Map (0 ~ {max_range_cm} cm)", fontsize=16)

 
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (cm)', fontsize=14)

    h, w = depth_map_cm.shape
    points_to_show = [
        #("",   int(w*0.5625), int(h*0.5625)),
    ]
    for name, x, y in points_to_show:
        d = depth_map_cm[y, x]
        if d > 0:
            plt.text(x, y, f"{name}\n{d:.1f}cm", color='white', fontsize=11,
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle="round,pad=0.3"),
                    ha='center')

    plt.subplot(2,2,4)
    # Grayscale pure numerical image (for post-processing or obstacle avoidance)
    plt.imshow(depth_map_cm, cmap='gray')
    plt.title("Depth Map (cm) - Grayscale", fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04, label='cm')
    plt.clim(0, max_range_cm)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/out_{file_serial_number}_{model_name}.jpg")
    if show : plt.show()

 


 

 
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def draw_labels(ax_or_img, labels, depth_map_cm, W, H, is_matplotlib=True):
    """
    The extracted plotting functions support Matplotlib (ax) or OpenCV (img).
    If is_matplotlib: True, plotting is done using ax; if False, plotting is done using cv2 on the img element.
    """
    for item in labels:
        name, cx, cy, bw, bh = item

        # Determine if normalized
        if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1:
            x_center = int(cx * W)
            y_center = int(cy * H)
            box_w = int(bw * W)
            box_h = int(bh * H)
        else:
            x_center = int(cx)
            y_center = int(cy)
            box_w = int(bw)
            box_h = int(bh)

        # bbox corners
        x1 = max(0, int(x_center - box_w / 2))
        y1 = max(0, int(y_center - box_h / 2))
        x2 = min(W-1, int(x_center + box_w / 2))
        y2 = min(H-1, int(y_center + box_h / 2))


        depth_cm = depth_map_cm[y_center, x_center]
        depth_text = "N/A" if depth_cm <= 0 else f"{depth_cm:.1f}cm"

        if is_matplotlib:
            # Matplotlib 
            ax = ax_or_img
            ax.add_patch(plt.Rectangle((x1, y1), box_w, box_h, edgecolor=(255/255, 20/255, 147/255), linewidth=2, fill=False))
            ax.text(x1, y1 - 10, f"{name} {depth_text}", color=(255/255, 20/255, 147/255), fontsize=15,
                    bbox=dict(facecolor='black', alpha=0.6, boxstyle="round,pad=0.3"))
        else:
            # OpenCV 
            img = ax_or_img
            cv2.rectangle(img, (x1, y1), (x2, y2), (147,20,255), 2)
            text = f"{name} {depth_text}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            cv2.putText(img, text, (x1 + 5, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (147,20,255), 2 )

def compose_label_with_depthmap(raw_img, disp, labels, num_disp, rectify_params,
                                model_name="",show=True, max_range_cm=110):
    if raw_img.shape[0] != disp.shape[0] or raw_img.shape[1] != disp.shape[1]:
        print("Image Size Not Equal")
    
 
    disp_vis = np.clip((disp) / num_disp, 0, 1)
    depth_map_mm = cv2.reprojectImageTo3D(disp, rectify_params["Q"])[:,:,2]
    depth_map_mm[disp <= 0] = 0
    depth_map_mm_clipped = depth_map_mm.copy()
    depth_map_mm_clipped[depth_map_mm > max_range_cm*10] = max_range_cm*10
    depth_map_cm = depth_map_mm_clipped / 10.0
    depth_norm = np.clip(depth_map_cm / max_range_cm, 0, 1)   
    depth_gray_8u = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(f"output/depth_gray_{model_name}.png", depth_gray_8u)
    H, W = depth_map_cm.shape

 
    img_copy = raw_img.copy()  # BGR 
    draw_labels(img_copy, labels, depth_map_cm, W, H, is_matplotlib=False)
    cv2.imwrite(f"output/fusioned_image_{model_name}.jpg", img_copy, [cv2.IMWRITE_JPEG_QUALITY, 100])   

 
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Left image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Left Image", fontsize=16)
    ax1.axis('off')
    draw_labels(ax1, labels, depth_map_cm, W, H, is_matplotlib=True)

    # Right image (depth)
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(depth_map_cm, cmap='gray', vmin=0, vmax=max_range_cm)
    ax2.set_title("Depth Map (cm)", fontsize=16)
    ax2.axis('off')

    # colorbar
    cax = inset_axes(ax2, width="4%", height="40%", loc='lower right', borderpad=1)
    plt.colorbar(im, cax=cax, label='cm')
    plt.tight_layout()
    plt.savefig(f"output/out_{model_name}.jpg")
    if show : plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_reconstruction(left_rect, disp, num_disp, rectify_params, model_name="", max_range_cm=1000, show=False, downsample=10):
    """
    Visualizes a 3D scene reconstruction from disparity map for robotic applications.
    Generates a 3D point cloud with colors from the rectified left image and saves a 2D projection image
    suitable for reports, showing the reconstructed environment from a perspective view.
    Added downsampling to reduce point density for cleaner visualization.
    Adjusted axes: Y as depth, Z as height (swapped original Y and Z).
    Oblique view angle for better perception.
    """
    # Compute 3D points (x, y, z) in mm
    points3d = cv2.reprojectImageTo3D(disp, rectify_params["Q"])
    
    # Filter valid points: disparity > 0 and depth within max range
    depth_mm = points3d[:,:,2]
    mask = (disp > 0) & (depth_mm > 0) & (depth_mm < max_range_cm * 10)
    
    # Downsample the mask and data to reduce points (e.g., every 'downsample' pixels)
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
        points3d = points3d[::downsample, ::downsample]
        left_rect = left_rect[::downsample, ::downsample]
        depth_mm = depth_mm[::downsample, ::downsample]
    
    # Extract valid 3D coordinates
    x = points3d[:,:,0][mask]  # X remains horizontal
    original_y = points3d[:,:,1][mask]  # Original Y (height)
    original_z = depth_mm[mask]  # Original Z (depth)
    
    # Swap for user request: Y as depth (original Z), Z as height (original Y, possibly inverted)
    y = original_z  # Y now as depth
    z = -original_y  # Z as height (invert Y to make positive upward, common in robotics)
    
    # Get corresponding colors from rectified left image (RGB normalized)
    colors = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)[mask] / 255.0
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=1, marker='.')
    
    # Set labels with adjusted axes
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (Depth, mm)', fontsize=12)
    ax.set_zlabel('Z (Height, mm)', fontsize=12)
    ax.set_title(f"3D Scene Reconstruction ({model_name})", fontsize=16)
    
    # Adjust view for oblique angle (diagonal perspective)
    ax.view_init(elev=30, azim=-60)  # Oblique view: 30° elevation, -60° azimuth for a slanted look
    
    # Tight layout and save high-quality image for report
    plt.tight_layout()
    plt.savefig(f"output/3d_reconstruction_{model_name}.png", dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)