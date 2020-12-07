import sys 
import os

# import custom modules 
sys.path.append(os.path.abspath(".."))
from config import BASE_PATH

def choice_input(display_text, choice_list, return_index = False):
    """ Displays the display_text and reads the users input while the answer is not in the accepted range. 
    """
    min_int_input = 0
    max_int_input = len(choice_list) - 1 
    input_int = min_int_input - 1 
    print("-"*60)
    while not (input_int <= max_int_input and input_int >= min_int_input):
        print(display_text)
        for index, choice_str in enumerate(choice_list):
            print("{}   --> {}".format(index, choice_str))
        input_int = int(input("Choose an option and type in the number: ")) 
    print("-"*60)
    if return_index:
        return input_int
    else:
        return choice_list[input_int]

custom_pretrained = choice_input(display_text="Use custom model as pretrained model?", choice_list=["yes", "no"], return_index=True)

# listing custom trained models 
if custom_pretrained == 0:
    print("Using custom model as pretrained model.")
    pretrained_config_path = BASE_PATH + '/trained_models/'
    
    possible_models = []
    for dir_files in os.listdir( pretrained_config_path ):
        if dir_files.count(".") == 0:
            possible_models.append(dir_files)

    if len(possible_models) <= 0:
        print("No model found. Can not use a custom pretrained model.")
        quit()
    
    pretrained_model_name = choice_input(display_text="Choose a model to train:", choice_list=possible_models)
    pretrained_model_path = os.path.join(BASE_PATH, 'trained_models/{}'.format(pretrained_model_name))

    import json
    with open(os.path.join(pretrained_model_path,'model_config.txt')) as json_file:
        cfg = json.load(json_file)
        p = cfg['model_config']
        chosen_model    = pretrained_model_name
        batch_size      = int(p['batch_size'])
        dataset_name    = p['dataset_name']
        num_done_steps  = int(p['num_done_steps'])
        num_eval_steps  = int(p['num_eval_steps'])

    num_steps_to_do = int(input("Number of epochs: "))
    num_steps = num_steps_to_do - num_done_steps
    if num_steps <= 0:
        print("{} steps allready done. Increase the number of steps.".format(num_done_steps))
        quit()

elif custom_pretrained == 1:
    possible_models = [
                       'efficientdet-d0', 
                       'efficientdet-d1', 
                       'efficientdet-d2', 
                       'efficientdet-d3',
                       
                       'ssd_mobilenet_v2',
                       ]

    chosen_model = choice_input("Choose a pretrained model to use:", possible_models)
    batch_size = 12
    dataset_name = "dataset-v1-dih4cps"
    num_steps = 10000 
    num_eval_steps = 500 

import datetime
timestamp = datetime.datetime.now()
timestamp_str = "_{}-{}-{}".format(timestamp.year, 
                                    timestamp.month, 
                                    timestamp.day)
output_model_name = chosen_model + timestamp_str

print("Selected Model: {}".format(chosen_model))
print("Selected Dataset: {}".format(dataset_name))
print("Batch size: {}".format(batch_size))
print("Number of steps: {}".format(num_steps))
print("Number of steps before evaluation: {}".format(num_eval_steps))
print("Output model name/dir: {}".format(output_model_name))

## install tensorflow dependencies 

tf_models_path = os.path.join(BASE_PATH, "tf2_object_detection_API", "models")
if not os.path.exists(tf_models_path):
    print("Can not find tensorflow models dir.")
    quit()

sys.path.append(tf_models_path)
sys.path.append(os.path.join(tf_models_path, "research"))
sys.path.append(os.path.join(tf_models_path, "research", "slim"))

## Import packages
import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

## prepare dataset
import boto3 
from getpass import getpass

ACCESS_KEY      = "minio"
CLOUD_URL       = "https://minio.dih4cps.swms-cloud.com:9000/"
SECRET_KEY      = "miniostorage" #getpass("Enter the secret key for {}: ".format(ACCESS_KEY))

def download_dataset(dataset_name, dataset_version):
    dataset_path = os.path.join(BASE_PATH, 'datasets')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    dataset_path = os.path.join(dataset_path, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    dataset_path = os.path.join(dataset_path, dataset_version)
    
    if os.path.exists(dataset_path):
        # dataset already downloaded
        return dataset_path
    
    ## if dataset not downloaded yet, download the selected dataset version
    os.mkdir(dataset_path)

    files_to_download = ['{}/train.record'.format(dataset_version), 
                         '{}/test.record'.format(dataset_version), 
                         '{}/images.txt'.format(dataset_version),
                         'labelmap.pbtxt']

    s3_client = boto3.client('s3',
            aws_access_key_id = ACCESS_KEY,
            aws_secret_access_key = SECRET_KEY,
            endpoint_url=CLOUD_URL, 
            verify=True,
            config=boto3.session.Config(signature_version='s3v4'))

    for f in files_to_download:
        s3_client.download_file(dataset_name,           # bucket name
                f,                                      # file key from cloud
                dataset_path + '/' + f.split("/")[-1])  # local filename
                                
        
    return dataset_path

def get_possible_dataset_versions(dataset_name):
    versions = []
    s3_resource = boto3.resource('s3',
            aws_access_key_id = ACCESS_KEY,
            aws_secret_access_key = SECRET_KEY,
            endpoint_url=CLOUD_URL, 
            verify=True,
            config=boto3.session.Config(signature_version='s3v4'))
    bucket = s3_resource.Bucket(dataset_name)

    for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            version = object_name.split("/")[0]
            if versions.count(version) <= 0 and version.count("version") > 0:
                versions.append(version)

    return versions 

possible_versions = get_possible_dataset_versions(dataset_name)
if len(possible_versions) <= 0:
    print("No dataset version available, check the dataset in the minio cloud.")
dataset_version = choice_input("Choose a dataset version:", possible_versions)
dataset_path = download_dataset(dataset_name, dataset_version)
test_record_fname = dataset_path + '/test.record'
train_record_fname = dataset_path + '/train.record'
label_map_pbtxt_fname = dataset_path + '/labelmap.pbtxt'
print(dataset_path)

## Configure Custom TensorFlow2 Object Detection Training Configuration
##change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz', 
        'batch_size': 16
    },
} # if you add a new model, also add the model short name to selection dropdown 


#in this tutorial we implement the lightweight, smallest state of the art efficientdet model
#if you want to scale up tot larger efficientdet models you will likely need more compute!

pretrained_model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

#download pretrained weights
os.chdir(BASE_PATH)
pretrained_path = os.path.join("tf2_object_detection_API","pretrained_models")
if not os.path.exists(pretrained_path):
    os.mkdir(pretrained_path)
os.chdir(pretrained_path)

import tarfile
import os

if not os.path.exists(pretrained_checkpoint):
    download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
    os.system("wget {}".format(download_tar))
else:
    print("Already downloaded {}".format(pretrained_checkpoint))

if not os.path.exists(pretrained_checkpoint.split(".")[0]):
    tar = tarfile.open(pretrained_checkpoint)
    tar.extractall()
    tar.close()
else:
    print("Already extracted {}".format(pretrained_checkpoint.split(".")[0]))
    
#download base training configuration file
import os
if not os.path.exists(base_pipeline_file):
    download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + base_pipeline_file
    os.system("wget {}".format(download_config))
else:
    print("Already downloaded {}".format(base_pipeline_file))

#prepare
pipeline_fname = os.path.join(pretrained_path,  base_pipeline_file)
fine_tune_checkpoint = os.path.join(pretrained_path, pretrained_model_name, "checkpoint", "ckpt-0")
def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)


def to_path_string(file_path):
    parts = []
    head = file_path
    while not head == "":
        head, tail = os.path.split(head)
        if tail == "":
            parts.append("C:")
        #if tail.lower() == "objectdetection":
            break
        parts.append(tail)
    path_str = ""
    for i, part in enumerate(parts[::-1]):
        if i > 0:
            path_str += "/"
        path_str += str(part)
    return path_str

#write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file
import re

os.chdir(BASE_PATH)
if not os.path.exists("created_models"):    
    os.mkdir("created_models")
os.chdir("created_models")
if not os.path.exists(output_model_name):  
    os.mkdir(output_model_name)
os.chdir(output_model_name)

print('writing custom configuration file')
with open(os.path.join(BASE_PATH, pipeline_fname)) as f:
    s = f.read()
created_config_path = os.path.join(BASE_PATH, "created_models", output_model_name, "pipeline_file.config")

with open(created_config_path, 'w') as f:
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(to_path_string(fine_tune_checkpoint)), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(to_path_string(train_record_fname)), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(to_path_string(test_record_fname)), s)
    
    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(to_path_string(label_map_pbtxt_fname)), s)
     
    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    
    #fine-tune checkpoint type
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
    f.write(s)

os.chdir(BASE_PATH)
pipeline_file = './created_models/{}/pipeline_file.config'.format(output_model_name)
model_dir = './created_models/{}/training'.format(output_model_name)

print("Training model with")
print("config:             {}".format(pipeline_file))
print("model directory:    {}".format(model_dir))
print("Number of steps:    {}".format(num_steps))
print("Number eval steps:  {}".format(num_eval_steps))

os.system("python tf2_object_detection_API/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={} \
    --model_dir={} \
    --alsologtostderr \
    --num_train_steps={} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={}".format(pipeline_file, model_dir, num_steps, num_eval_steps))