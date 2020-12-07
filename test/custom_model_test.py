"""
custom_model_test: This module can be used to test a trained model. 

Make sure you have added a video file for testing in a directory test_videos/ and adapted the filename. See declaration of video_file_path.

Also make sure you extracted the model graph to a directory called created_model_graph. 

The other files (utils and label map) will be added by including the path to these files of the object detection api. Also it is possible to add these files in the directory utils/ and data/.  

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v0.0.1
@license:   ...

VERSION HISTORY
Version:    (Author) Description:                                   Date:
v0.0.1      (AS) First initialize. Added important configurations.  14.10.2020\n
"""


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


file_dir_path = os.path.dirname(__file__)

sys.path.append(os.path.join(file_dir_path, "../python_modules"))
from config import BASE_PATH, DATASET_PATH

if not os.path.exists(os.path.join(file_dir_path, "utils")):
    sys.path.append(os.path.join(BASE_PATH, "tf2_object_detection_API/models/research/object_detection"))
else:
    print("using local utils dir.")

sys.path.append(os.path.join(BASE_PATH, "tf2_object_detection_API/models/research"))
from utils import label_map_util
from utils import visualization_utils as vis_util


import cv2
# 10 s
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_06-22-29.avi"  # 0 / 1
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_06-12-29.avi"  # 2 / 5
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_06-47-44.avi"  # 1 / 2
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_06-57-32.avi"  # 2 / 2
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_06-32-29.avi"  # 0 / 3
video_file_path = "./test_videos/test1_cronjob_2020-10-06_02-57-48.avi"
# 30 s
#video_file_path = "./test_videos/test1_cronjob_2020-10-13_07-30-42.avi"  # 1 / 6 

cap = cv2.VideoCapture(video_file_path)   # if you have multiple webcams change the value to the correct one

# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.

MODEL_NAME = 'tf_API_data1_v01'
MODEL_GRAPH = MODEL_NAME + '_graph_1' 

# MODEL_FILE = MODEL_NAME + '.tar.gz'   # these lines not needed as we are using our own model
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(file_dir_path, MODEL_GRAPH, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
DATASET_NAME = "dataset-v1-dih4cps"
PATH_TO_LABELS = os.path.join(DATASET_PATH, DATASET_NAME, 'labelmap.pbtxt')

NUM_CLASSES = 1  # we are only using one class in this example (cloth_mask)

# we don't need to download model since we have our own
# ## Download Model

# In[5]:
#
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

""" testing with images 
# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
"""
def calculate_box_intense(image, box):
    """
    x_1 = int(box[0] * image.shape[0])
    x_2 = int(box[2] * image.shape[0])
    y_1 = int(box[1] * image.shape[1])
    y_2 = int(box[3] * image.shape[1])
    box_image = image[x_1:x_2, y_1:y_2, :]

     
    blurred = cv2.GaussianBlur(box_image, (5, 5), 0)

    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    
    def filterOutSaltPepperNoise(edgeImg):
        # Get rid of salt & pepper noise.
        count = 0
        lastMedian = edgeImg
        median = cv2.medianBlur(edgeImg, 3)
        while not np.array_equal(lastMedian, median):
            # get those pixels that gets zeroed out
            zeroed = np.invert(np.logical_and(median, edgeImg))
            edgeImg[zeroed] = 0

            count = count + 1
            if count > 70:
                break
            lastMedian = median
            median = cv2.medianBlur(edgeImg, 3)

    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)
    
    def findSignificantContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Find level 1 contours
        level1Meta = []
        for contourIndex, tupl in enumerate(hierarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl.copy(), 0, [contourIndex])
                level1Meta.append(tupl)
    
        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for tupl in level1Meta:
            contourIndex = tupl[0]
            contour = contours[contourIndex]
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area, contourIndex])
            
        contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
        largestContour = contoursWithArea[0][0]
        return largestContour
            
    contour = findSignificantContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(box_image)
    
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    cv2.imshow('contour', contourImg)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

            
    
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x_1 = int(box[0] * gray_image.shape[0])
    x_2 = int(box[2] * gray_image.shape[0])
    y_1 = int(box[1] * gray_image.shape[1])
    y_2 = int(box[3] * gray_image.shape[1])

    box_image = image[x_1:x_2, y_1:y_2, :]
    box_image_gray = gray_image[x_1:x_2, y_1:y_2]
    
    ret, box_image_gray = cv2.threshold(box_image_gray, 128, 255, cv2.THRESH_BINARY)
    #print(len(box_image_gray.flatten()))
    print(np.count_nonzero(box_image_gray.flatten() > 0))
    cv2.imshow("Box", box_image)
    cv2.imshow("Box Thresh", box_image_gray)
    cv2.waitKey(25) 
        
    import time
    time.sleep(1)
    
# In[10]:

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while cap.isOpened():
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [ boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            boxes = np.squeeze(boxes)
            """
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            """

            count = 0 
            for index, score in enumerate(np.squeeze(scores)):
                if score >= 0.5:
                    calculate_box_intense(image_np, boxes[index])
                    count += 1
            """
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            """
            import time
            time.sleep(1)

        cap.release()

