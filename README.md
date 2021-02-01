# Object Detection by Schiller 
This Repository is used to train and process object detection.

## Structure:
ObjectDetection 
|-- datasets (includes dataset files for training and model evaluation, see datasets/README.md)
|   
|-- tf2_object_detection_API
|   |-- models (TensorFlow repository)
|   '-- pretrained_models (downloaded pretrained models)
|
|-- yolov5 (ultralytics repository)
|
|-- .git/.gitignore
|-- LICENSE (MIT)
'-- README.md (this file)

## Informations about used repositories 
To train models from the tensorflow object detection API the tensorflow/models repository is cloned from https://github.com/tensorflow/models.
´git clone git@github.com:tensorflow/models.git´
Also the the repository from the yolov5 is used to train yolov5 models (https://github.com/ultralytics/yolov5).
´cd tf2_object_detection_API´
´git clone git@github.com:ultralytics/yolov5.git´

To evaluate models the Object-Detection-Metrics repository is cloned and adapted (https://github.com/rafaelpadilla/Object-Detection-Metrics/).


