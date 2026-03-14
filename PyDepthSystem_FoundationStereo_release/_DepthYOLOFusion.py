import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyStereo.StereoLib import *
from PyStereo.StereoMatchers import *
from PyStereo.DetectorLib import *
from ultralytics import YOLO
import shutil
import os





serial_number = ""
DATASET = "input/Expirement01"

left_img  = cv2.imread(f"{DATASET}/stereoLeft/left_{serial_number}.png")
right_img = cv2.imread(f"{DATASET}/stereoRight/right_{serial_number}.png")

raw = left_img.copy()


#  YOLOv11 detection 

yolo_model = YOLO("yolov8n-oiv7.pt")

results = yolo_model(raw)[0]

labels = []

for box in results.boxes:

    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

    labels.append({
        "bbox":[x1,y1,x2,y2],
        "class_name":results.names[cls_id],
        "conf":conf
    })


# stereo rectify 

stereo_rectify_params = load_stereo_camera_params()

left_img , right_img  = rectify_stereo_images(
    left_img,
    right_img,
    rectify_params=stereo_rectify_params,
    show=False
)


# stereo matching 
def use_FS(): 
    fs_stereo = FoundationStereoV1(device='cuda')
    return fs_stereo.stereo_matching, "FoundationStereoV1" 
model_func , model_name = use_FS()

disp = model_func(left_img, right_img)

compose_label_with_depthmap(
    raw,
    disp,
    labels,
    FoundationStereoV1.max_disp,
    stereo_rectify_params,
    model_name=model_name,
    max_range_cm=200,
    show=False
)