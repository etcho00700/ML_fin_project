# Import the required modules
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 8
from skimage import img_as_float
from skimage import exposure
import plotly.graph_objects as go

import os
import glob
import random
from skimage import io # To preprocess the images
from distutils.file_util import copy_file
import seaborn as sns
import cv2
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from skimage.transform import rescale
from keras_preprocessing.image import ImageDataGenerator

import warnings
warnings.simplefilter('ignore')

# COVID and Normal dataset directory
CLASSES = ["COVID",  "Normal"]

#Image augmentation process:
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    horizontal_flip = True,
    vertical_flip = False,
    shear_range = 0.2,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    
    #split dataset to training(80%) and validation(20%):
    validation_split = 0.2
)

test_data_generator = ImageDataGenerator(
    rescale=1./255,
)

# Training dataset and Validation dataset:
train_data = train_datagen.flow_from_directory(
    directory='./small_Dataset/Train',
    target_size=(299, 299),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset='training',
    classes=CLASSES
    )
val_data = train_datagen.flow_from_directory(
    directory='./small_Dataset/Train',
    target_size=(299, 299),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset='validation',
    classes=CLASSES
    )
# Testing dataset:
test_data = test_data_generator.flow_from_directory(
    directory='./small_Dataset/Test/',
    target_size=(299, 299),
    class_mode='binary',
    batch_size=16,
    shuffle=True
)

#Using sequential model:
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(299, 299, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile our model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy']
              )

# Training process:
start_time = datetime.datetime.now()

number_epochs = 20
history = model.fit(train_data,
                    epochs=number_epochs,
                    validation_data=val_data,
                    verbose=1
                    )

end_time = datetime.datetime.now()
print(f'Total Training Time for Sequential Model: {end_time - start_time}')

pd.DataFrame(history.history).plot()
plt.show()

test_accuracy = model.evaluate(test_data)

print("Accuracy of Sequential Model: " + str(round(test_accuracy[1] * 100, 2)) + "%")