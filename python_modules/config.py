# import the necessary packages
import os

def get_base_path():
    path = os.path.dirname(__file__)
    while len(os.path.split(path)) >= 1 and not os.path.split(path)[-1].lower() == "objectdetection":
        path = os.path.dirname(path)
    return path
BASE_PATH = get_base_path()

PYTHON_PATH = os.path.join(BASE_PATH, "python_modules")

DATASET_PATH = os.path.join(BASE_PATH, "datasets")

# set true if the images of the selectiveSearch need to be updated 
UPDATE_IMAGES = False 

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)
# define the path to the output model and label binarizer
MODEL_PATH = "shrimp_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99