import tensorflow as tf
import numpy as np
import tarfile
import zipfile
import cv2
import sys
import os
from matplotlib import pyplot as plt
from collections import defaultdict
import six.moves.urllib as urllib
from io import StringIO
from PIL import Image

file_dir_path = os.path.dirname(__file__)

sys.path.append(os.path.join(file_dir_path, ".."))
from config import BASE_PATH, DATASET_PATH

if not os.path.exists(os.path.join(file_dir_path, "utils")):
    sys.path.append(os.path.join(BASE_PATH, "tf2_object_detection_API/models/research/object_detection"))
else:
    print("using local utils dir.")

sys.path.append(os.path.join(BASE_PATH, "tf2_object_detection_API/models/research"))
from utils import label_map_util
from utils import visualization_utils as vis_util


class TfModel():

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
    def __init__(self):
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
        #
        # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

        DATASET_NAME = "dataset-v1-dih4cps"
        labelmap_path = os.path.join(DATASET_PATH, DATASET_NAME, 'labelmap.pbtxt')

        # MODEL_NAME = 'efficientdet-d0_2020_11_10'
        # model_path = os.path.join(BASE_PATH, "trained_models", MODEL_NAME)

        # checkpoint_path = os.path.join(model_path, "checkpoint")
        # saved_model_path = os.path.join(model_path, 'saved_model', 'saved_model.pb')
        
        model_path = os.path.join(os.path.dirname(__file__), "tf_model")
        checkpoint_path = os.path.join(model_path, "variables")
        saved_model_path = os.path.join(model_path, 'saved_model.pb')
        print(saved_model_path)

        NUM_CLASSES = 1  # we are only using one class in this example (cloth_mask)

        ###### FUNKTION EINBINDEN: DATASETHANDLER LOAD DATASET INFORMATIONS

        # load detection graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(saved_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

        label_map = label_map_util.load_labelmap(labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        print(self.category_index)

    def image_detection_test(self):
        """ testing with images 
        # ## Helper code

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)


        # # Detection
        # For the sake of simplicity we will use only 2 images:
        # image1.jpg
        # image2.jpg
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        """
        pass

    def predict(self, image_np):
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [ boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                boxes = np.squeeze(boxes)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                count = 0 
                for index, score in enumerate(np.squeeze(scores)):
                    if score >= 0.5:
                        #calculate_box_intense(image_np, boxes[index])
                        count += 1
                print(count)

                return boxes, scores, self.category_index


import onnx2keras
from onnx2keras import onnx_to_keras
import tensorflow.keras
import onnx

if __name__ == '__main__':
    # MODEL_NAME = 'efficientdet-d0_2020_11_10'
    # model_path = os.path.join(BASE_PATH, "trained_models", MODEL_NAME)
    # saved_model_path = os.path.join(model_path, 'saved_model')
    
    """
    onnx_model = onnx.load('best.onnx')
    k_model = onnx_to_keras(onnx_model, ['images'])
    k_model.summary()
    keras.models.save_model(k_model,'kerasModel.h5',overwrite=True,include_optimizer=True)
    """
    """
    model_path = os.path.join(os.path.dirname(__file__), "tf_model")
    saved_model_path = os.path.join(model_path)
    model = tf.keras.models.load_model(saved_model_path)
    #model = tf.saved_model.load(saved_model_path)
    # model = TfModel()
    test_img = os.path.join(BASE_PATH, "testing", "test1_cronjob_2020-10-01_12-39-44_51.png")
    img = np.array(cv2.imread(test_img))
    print(model.summary())
    """
#= TfModel()
