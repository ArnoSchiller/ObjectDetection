import tensorflow as tf 
import numpy as np
import tarfile
import zipfile
import glob
import cv2
import sys
import os
from matplotlib import pyplot as plt
from collections import defaultdict
import six.moves.urllib as urllib
from imageio import imread
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


class Model():

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
    def __init__(self, model_name, img_size):
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
        #
        # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
        self.img_size = img_size
        DATASET_NAME = "dataset-v1-dih4cps"
        labelmap_path = os.path.join(DATASET_PATH, DATASET_NAME, 'labelmap.pbtxt')

        # MODEL_NAME = 'efficientdet-d0_2020_11_10'
        # model_path = os.path.join(BASE_PATH, "trained_models", MODEL_NAME)

        # checkpoint_path = os.path.join(model_path, "checkpoint")
        # saved_model_path = os.path.join(model_path, 'saved_model', 'saved_model.pb')
        

        model_path = os.path.join("..", "trained_models", model_name, "saved_model")
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

    def load_images(self, images_path):
        files = glob.glob(os.path.abspath(images_path) + "*.png")

        data = []
        for f in files:
            img = imread(f)
            data.append((f, img))

        return data

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


def detect(images_dir, model_name, img_size, original_img_size, txt_out_path=None, xml_out=False, dict_out=False, all_classes=None, visualisation=True):
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

    model = Model(model_name, img_size)

    data = model.load_images(images_dir)

    for idx, (path, img) in enumerate(data):
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

            with open(os.path.join(txt_out_path, file_name + ".txt"), "w") as f:
                for idx, gt_box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = gt_box
                    class_name = classes[idx]
                    s = scores[idx]
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
