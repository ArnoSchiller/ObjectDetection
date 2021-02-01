import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import os

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont


def plot_boxes_on_image(image, boxes, classes=None, mode="xyrb", relative=False, save_path=None):
    """
    mode : 
        tf = [ymin, xmin, ymax, xmax] --> will be sorted to xyrb
        xyrb = [x_min, y_min, x_max, y_max]
        xywh = [x_center, y_center, width, height]
    """
    if mode == "tf":
        #       xmin    ymin    xmax    ymax
        for idx, box in enumerate(boxes):
            boxes[idx] = [box[1], box[0], box[3], box[2]]
        mode = "xyrb" 
    
    if relative:
        (img_h, img_w) = image.shape[:2]
        for idx, box in enumerate(boxes):
            boxes[idx] = [int(box[0]*img_w), int(box[1]*img_h), int(box[2]*img_w), int(box[3]*img_h)]

    if mode == "xyrb":
        for idx, box in enumerate(boxes):
            x_c = int((box[2] + box[0]) / 2)     # x_center
            y_c = int((box[3] + box[1]) / 2)     # y_center
            box_w = int(box[2] - box[0])         # box width
            box_h = int(box[3] - box[1])         # box height 
            
            boxes[idx] = [x_c, y_c, box_w, box_h]
        print(boxes)
        mode = "xywh"
    
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    for idx, box in enumerate(boxes):
        # Create a Rectangle patch [coords like:  (x_min,y_max),w,h] 
        #                                         buttom-left 
        xmin = box[0] - int(box[2]/2)
        ymin = box[1] - int(box[3]/2)
        rect = patches.Rectangle((xmin, ymin), box[2], box[3], linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    if not save_path is None:
        plt.savefig(save_path)

    plt.show()


def load_annotation_from_txt(path):

    boxes = []
    classes = []
    scores = []

    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        annotation = line.split(" ")
        classes.append(annotation[0])

        if len(annotation) > 5:
            scores.append(float(annotation[1]))
            i = 2
        elif len(annotation) == 5:
            scores.append(-1)
            i = 1
        else:
            continue
        boxes.append([float(annotation[i]), float(annotation[i+1]), float(annotation[i+2]), float(annotation[i+3])])

    return boxes, classes, scores

def load_image_to_np_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

imgs = [
    "test1_cronjob_2020-09-30_12-55-26_144_gray.png", 
    "test1_cronjob_2020-10-01_08-49-20_189_gray.png", 
    "test1_cronjob_2020-10-01_09-09-15_135_gray.png",
    "test1_cronjob_2020-10-07_05-40-02_68_gray.png",
    "test1_cronjob_2020-10-01_09-32-52_93_gray.png",
    "test1_cronjob_2020-10-02_14-15-38_173_gray.png",
    "test1_cronjob_2020-10-02_20-45-30_172_gray.png",
    "test1_cronjob_2020-10-06_01-47-37_48_gray.png",
    ]
imgs_path = os.path.join("..", "datasets", "dataset-v1-dih4cps", "version_1_gray", "voc", "test")
model_name = "yolov5l_dsv1_gray"
gt_path = os.path.join(".", "Object-Detection-Metrics", model_name + "_gt")
pred_path = os.path.join(".", "Object-Detection-Metrics", model_name + "_pred")
result_path = os.path.join(".", "Object-Detection-Metrics", "results_" + model_name)

# pick random image 
image_files = [os.path.join(imgs_path, img) for img in imgs]
rand_idxs =  range(0, len(image_files)) # np.random.randint(0, len(image_files)-1, 10)

# pick paths and add label path
img_paths = []
label_paths = []
for idx in rand_idxs:
    path = image_files[idx]
    img_paths.append(path)
    _, filename = os.path.split(path)
    path = os.path.join(pred_path, filename.split(".")[0]+".txt")
    label_paths.append(path) 

for idx, path in enumerate(img_paths):
    im = load_image_to_np_array(path)##np.array(Image.open(path), dtype=np.uint8)
    annotation = load_annotation_from_txt(label_paths[idx])

    print(annotation[1])
    fname = os.path.basename(path)
    save_path = os.path.join(result_path, fname)

    if annotation:
        plot_boxes_on_image(image=im, boxes=annotation[0],mode="xyrb", relative=False, save_path=save_path)

