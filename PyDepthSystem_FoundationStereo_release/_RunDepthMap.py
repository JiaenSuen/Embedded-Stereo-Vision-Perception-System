import cv2
import numpy as np
from PyStereo.StereoLib import *
from PyStereo.StereoMatchers import *
from PyStereo.DetectorLib import *
import shutil
import time
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# get every path in a folder
MAX_RANGE_CM = 100
input_dir = "input/Expirement01"
serial_numbers_list = os.listdir(f"{input_dir}/stereoLeft")
serial_numbers_list = [sn.split("_")[-1].split(".")[0] for sn in serial_numbers_list]



stereo_rectify_params = load_stereo_camera_params()






 
def use_FS(): # -> Return Matching Function, Model Name
    fs_stereo = FoundationStereoV1(device='cuda')
    return fs_stereo.stereo_matching, "FoundationStereoV1" 
model_func , model_name = use_FS()



# output
if not os.path.exists("output"): os.makedirs("output")
out_dir = f"output/{input_dir[input_dir.rfind('/'):]}_DepthMaps_{model_name}"
if os.path.exists(out_dir): shutil.rmtree(out_dir);os.makedirs(out_dir)
else: os.makedirs(out_dir)
 

 
 
for serial_number in serial_numbers_list:
    left_img  = cv2.imread(f"{input_dir}/stereoLeft/left_{serial_number}.png")
    right_img = cv2.imread(f"{input_dir}/stereoRight/right_{serial_number}.png")
    
    left_img , right_img  = rectify_stereo_images(left_img,right_img,rectify_params=stereo_rectify_params,show=False) 

    # try load yolo labels
    
    
    
 


    start_time =  time.time()
    disp   = model_func(left_img, right_img)
    end_time = time.time()
    print(f"{model_name} took {end_time - start_time:.2f} seconds.")
 
 


    print(f"Disparity map shape and size from {model_name}: {disp.shape}, {disp.nbytes/1024/1024:.2f} MB")
    visualize_depth_map(
        left_img,
        disp,
        FoundationStereoV1.max_disp,
        stereo_rectify_params,
        model_name=model_name,
        file_serial_number=serial_number,
        show=False,
        max_range_cm=MAX_RANGE_CM,
        output_dir=out_dir
    )

    
    