import numpy as np
import torch
import cv2
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_PATH
from datasetHandling.dataset_handler import DatasetHandler

# add yolov5 repo to path
sys.path.append(os.path.join(BASE_PATH, "yolov5"))

# and import modules from yolov5
from models.experimental import attempt_load
from utils.general import check_img_size, set_logging, non_max_suppression
from utils.torch_utils import select_device
from utils.datasets import LoadImages


class Model():
    """
    This class can be used to load and run models trained with pytorch. 

    Needs:
        - pytorch model file, called best.pt into the directory pt_model -> update to trained model 
        - A clone of the yolov5 repo (location: Objectdetection/yolov5) 
        - requirements installed, see yolov5 repo (torchvision --> pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html)

    @authors:   Arno Schiller (AS)
    @email:     schiller@swms.de
    @version:   v0.0.1
    @license:   ...

    VERSION HISTORY
    Version:    (Author) Description:                               Date:
    v0.0.1      (AS) First initialize. Implemented using detect.py  14.10.2020\n
                    from https://github.com/ultralytics/yolov5.
    """

    def __init__(self, weights, img_size, device=''):
        print("Init")
        
        self.augment = False
        self.agnostic_nms = False
        self.classes = [0]
        self.conf_thres = 0
        self.iou_thres = 0.5

        self.class_names = ["shrimp"]
        
        self.class_names = [
            'shrimp_head_near', 'shrimp_head_middle', 'shrimp_head_far', 'shrimp_tail_near', 'shrimp_tail_middle', 'shrimp_full_far', 'shrimp_full_near', 'shrimp_full_middle', 'shrimp_tail_far', 'shrimp_part_near', 'shrimp_part_middle', 'shrimp_part_far']
        
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        #self.imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        #if self.half:
        #    self.model.half()  # to FP16
        
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        #img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  
        #_ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None
    
    def predict(self, image_np, original_img_size=None):

        if original_img_size is None:
            original_img_size = image_np.shape[::-1]

        img = torch.from_numpy(image_np).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.augment)[0]
    
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Process detections
        boxes = []
        scores = []
        classes = []

        for i, det in enumerate(pred):
            if not det.size()[0] > 0:
                return boxes, scores, classes, original_img_size

            [xmin, ymin, xmax, ymax, confidence, class_idx] = det[0]
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                print(n)
            if confidence >= self.conf_thres:
                # normalise boxes
                # xmin = xmin/self.imgsz * original_img_size[0]
                # ymin = ymin/self.imgsz * original_img_size[1]
                # xmax = xmax/self.imgsz * original_img_size[0]
                # ymax = ymax/self.imgsz * original_img_size[1]
                        
                boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                scores.append(float(confidence))
                classes.append(self.class_names[int(class_idx)])

        return boxes, scores, classes, original_img_size

def detect(images_dir, weights, img_size, original_img_size, txt_out_path=None, xml_out=False, dict_out=False, all_classes=None, visualisation=False):
    if xml_out:
        from helpers.xml_handler import XMLHandler
        xml_handler = XMLHandler()
    
    if dict_out:
        if all_classes == None:
            print("Add a list of possible classes for dict output.")
            return {}
        output_dict = {}
        for class_name in all_classes:
            output_dict[class_name] = {}

    out_file_path = ""

    data = LoadImages(images_dir, img_size=img_size)

    model = Model(weights, img_size)

    for idx, (path, img, _, _) in enumerate(data):
        boxes, scores, classes, img_size = model.predict(img, original_img_size=original_img_size)

        if not txt_out_path is None:
            _, file_name = os.path.split(path)
            file_name = file_name.split(".")[0]
            # write a txt for every file including the gts like 
            # {class} {xmin} {ymin} {xmax} {ymax}
            if idx+1 != len(data):
                print(f"    Writing file {idx+1} of {len(data)}", end='\r')
            else:
                print(f"    Writing file {idx+1} of {len(data)}")
            
            if len(boxes) > 1 or len(boxes) == 0:
                print(len(boxes))
            with open(os.path.join(txt_out_path, file_name + ".txt"), "w") as f:
                for box_idx, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = box
                    class_name = classes[box_idx]
                    s = scores[box_idx]
                    f.write(f"{class_name} {s} {xmin} {ymin} {xmax} {ymax}\n")

        if xml_out:
            path, fname = os.path.split(path) 
            path, _ = os.path.split(path) 
            path = os.path.join(path, "eval_images_pred")
            if not os.path.exists(path):
                os.mkdir(path)
            fname = fname.split(".")[0] + "_pred.xml"
            out_file_path = os.path.join(path, fname)
        
            xml_handler.create_detection_xml(output_filepath=out_file_path,
                                        img_filepath=path,
                                        boxes=boxes,
                                        scores=scores, 
                                        classes=classes,
                                        img_size=original_img_size) 

            return os.path.split(out_file_path)[0]

        if dict_out:
            for idx, box in enumerate(boxes):
                class_name = classes[idx]
                file_name = os.path.split(path)[-1]
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                if not list(output_dict[class_name].keys()).count(file_name)>0:
                    output_dict[class_name][file_name] = {}
                if not list(output_dict[class_name][file_name].keys()).count('boxes')>0:
                    output_dict[class_name][file_name]['boxes'] = [bbox]
                    output_dict[class_name][file_name]['scores'] = [scores[idx]]
                else:
                    output_dict[class_name][file_name]['boxes'].append(bbox)
                    output_dict[class_name][file_name]['scores'].append(scores[idx])

        if visualisation:
            img = np.array(cv2.imread(path))
            print(len(boxes))
            for idx, bb in enumerate(boxes):
                class_name = classes[idx]
                x,y = int(bb[0]), int(bb[1])
                w,h = int(bb[2] - bb[0]), int(bb[3] - bb[1])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,str(class_name),(x+w+10,y+h),0,0.3,(0,255,0))
            cv2.imshow("Show",img)
            cv2.waitKey()  
            cv2.destroyAllWindows()

    if dict_out:
        return output_dict


if __name__ == '__main__':

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__)
    , "best.pt")
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "yolov5l_dsv1_gray", "weights", "best.pt")
    IMG_SIZE = 640
    IMG_PATH = os.path.dirname(__file__)
    
    original_img_size = (640, 480, 3)

    pred = detect(IMG_PATH, WEIGHTS_PATH, IMG_SIZE, original_img_size, xml_out=False)
    print(pred)
    """

    m = Model(WEIGHTS_PATH, IMG_SIZE)
    data = LoadImages(IMG_PATH, img_size=IMG_SIZE)
    for _, img, _, _ in data:
        print(m.predict(img, original_img_size))
    """