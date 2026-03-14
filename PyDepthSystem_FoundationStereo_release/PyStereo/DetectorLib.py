



def load_yolo_labels_Form(label_path, img_w, img_h, class_names=None, return_pixel=False):
    labels = []

    try:
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue   

                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

 
                if class_names is not None:
                    class_name = class_names[class_id]
                else:
                    class_name = str(class_id)

                if return_pixel:
 
                    cx_p = cx * img_w
                    cy_p = cy * img_h
                    w_p = w * img_w
                    h_p = h * img_h
                    labels.append((class_name, cx_p, cy_p, w_p, h_p))
                else:
                    labels.append((class_name, cx, cy, w, h))

    except FileNotFoundError:
        print(f"Label file not found: {label_path}")

    return labels

def load_yolo_labels(file_path):
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue   
            
            parts = line.split()
            
 
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            labels.append([class_id, x_center, y_center, width, height])
    
    return labels
