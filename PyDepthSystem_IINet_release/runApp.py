import os
import cv2
import numpy as np
import time
import threading
import queue


from PyStereo.StereoLib import *
from PyStereo.StereoMatchers import *


CAMERA_ID = 1                 
TARGET_FPS = 2                 
DISPLAY_SCALE = 0.5              

os.environ["DISPLAY"] = ":0"


model_func, model_name = IINetv1.stereo_matching, "IINet"
stereo_rectify_params = load_stereo_camera_params()


print("Opening camera with stable OpenCV V4L2 (3200x1200)...")
cap = cv2.VideoCapture(CAMERA_ID)#, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

if not cap.isOpened():
    raise RuntimeError("Cannot open camera! Check cable and /dev/video1")

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera ready! Real resolution = {actual_w}x{actual_h}")



frame_queue = queue.Queue(maxsize=1)
running = True

def capture_thread():
    global running
    print("Capture thread started (aggressive grab mode)...")
    fail_count = 0

    while running:

        grabbed = False
        for _ in range(6):               
            if cap.grab():
                grabbed = True
            else:
                break

        if grabbed:
            ret, frame = cap.retrieve()
            if ret:
                try:
                    frame_queue.put_nowait(frame.copy()) 
                except queue.Full:
                    pass
                fail_count = 0
            else:
                fail_count += 1
        else:
            fail_count += 1

        if fail_count > 10:
            print("WARNING: Camera stalled 10+ times - check USB cable/power")

        time.sleep(0.001)  

 
capture_thread = threading.Thread(target=capture_thread, daemon=True)
capture_thread.start()

print("Stereo demo started - Aggressive Thread Fix (should never stall again)")

 
def disparity_to_colormap(disp):
    """Convert disparity map to JET colormap"""
    disp = np.nan_to_num(disp)
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

 
frame_count = 0

while True:
    start_time = time.time()


    try:
        frame = frame_queue.get(timeout=2.0)
    except queue.Empty:
        print("Warning: No new frame (rare now)")
        continue

    frame_count += 1

    # Debug only first 5 frames
    if frame_count <= 5:
        print(f"Frame {frame_count} | Raw: {frame.shape}")

    # Split left/right (3200x1200 side-by-side)
    mid = frame.shape[1] // 2
    left_raw = frame[:, :mid]
    right_raw = frame[:, mid:]

    # Rectify
    left_img, right_img = rectify_stereo_images(
        left_raw, right_raw,
        rectify_params=stereo_rectify_params,
        show=False
    )

    # Run model (IINet)
    disp = model_func(left_img, right_img)

    # Visualize
    depth_vis = disparity_to_colormap(disp)

    # Resize & combine
    left_show = cv2.resize(left_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    depth_show = cv2.resize(depth_vis, (left_show.shape[1], left_show.shape[0]))
    display = np.hstack((left_show, depth_show))

    # FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(display, f"{model_name} FPS:{fps:.2f} ", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("Stereo Depth Demo - AGGRESSIVE GRAB FIX", display)

    if cv2.waitKey(int(1000 / TARGET_FPS)) & 0xFF == 27:
        break

#  CLEANUP 
running = False
cap.release()
cv2.destroyAllWindows()
print("Stereo demo stopped")