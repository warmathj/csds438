# -*- coding: utf-8 -*-
"""csds438_final_project
Author: Long Phan 
id: lnp26
"""

import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm 
from tensorflow.keras import datasets, layers, models
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import argparse


# Transfer Learning
from tensorflow.keras.applications import ResNet152V2



parser = argparse.ArgumentParser(description='Args')
parser.add_argument('-data_path', dest='path', type=str, help='Data Path', default="../raw-img")
parser.add_argument('-epoch', dest='epoch', type=int, help='Training Epoch', default=5)
parser.add_argument('-image_size', dest='image_size', type=int, help='Image Size', default=128)
parser.add_argument('-model_name', dest='model_name', type=str, help='model_name', default="CNN")
parser.add_argument('-batch_size', dest='batch_size', type=int, help="Training batch size", default=32)
args = parser.parse_args()


print("==============NameSpaces==============")
print(args)

path = args.path
epoch = args.epoch
image_size = args.image_size
model_name = args.model_name
batch_size = args.batch_size

# Main Path
from tensorflow.python.client import device_lib


print("=============GPU Devices==============")
print(device_lib.list_local_devices())


# Get Class Names
class_names = sorted(os.listdir(path))
n_classes = len(class_names)
print(f"Class Names: \n{class_names}")
print(f"Total Number of Classes : {n_classes}")

# Initialize Generator 

print("Initialize Data Generator")
gen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=20, 
    validation_split=0.2)

print(f"Load Data from {path}")
# Load data
train_ds = gen.flow_from_directory(
    path, 
    target_size=(image_size,image_size), 
    class_mode='binary', 
    batch_size=batch_size, 
    shuffle=True, 
    subset='training')

valid_ds = gen.flow_from_directory(
    path, 
    target_size=(image_size,image_size), 
    class_mode='binary', 
    batch_size=batch_size, 
    shuffle=True, 
    subset='validation')

if "cnn" in model_name.lower():
  name = "CNN"

  model = models.Sequential()
  model.add(layers.Conv2D(image_size, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(GAP()),
  model.add(Dense(image_size, activation='relu'))
  model.add(Dropout(0.2)),
  model.add(Dense(n_classes, activation='softmax'))

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
  )

if "resnet" in model_name.lower():
  name = "ResNet152V2"

  base_model = ResNet152V2(include_top=False, input_shape=(image_size,image_size,3), weights='imagenet')
  base_model.trainable = False # Freeze the Weights


  resnet152V2 = Sequential([
      base_model,
      GAP(),
      Dense(image_size, activation='relu'),
      Dropout(0.2),
      Dense(n_classes, activation='softmax')
  ], name=name)

  # Compile
  resnet152V2.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
  )

  model = resnet152V2

# Train Model

cbs = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(f"output/{model_name}.h5", save_best_only=True)
]

print(f"======================Training {model_name} for {epoch} epochs======================")
history = model.fit(
    train_ds, validation_data=valid_ds,
    epochs=epoch, callbacks=cbs
)

