from imutils import paths
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# initialize the input dimensions to the network
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_DIMS, DATASET_PATH
from datasets.shrimp_dataset import ShrimpDataset

base_model=MobileNetV2(weights='imagenet',include_top=False) #imports the MobileNetV2 model and discards the last 1000 neuron layer.
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation for N classes

model=Model(inputs=base_model.input,outputs=preds) #specify the inputs and outputs

num_trainable_layers = 20
for layer in model.layers[:num_trainable_layers]:
    layer.trainable=False
for layer in model.layers[num_trainable_layers:]:
    layer.trainable=True
    
dataset = ShrimpDataset(dataset_name="dataset-v1-dih4cps", 					
						dataset_mode="selectiveSearch")
data, labels = dataset.load_dataset()
    
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()

EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

model.save("")