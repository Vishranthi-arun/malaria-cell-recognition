# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
Malaria dataset of 27,558 cell images with an equal number of parasitized and uninfected cells. A level-set based algorithm was applied to detect and segment the red blood cells. The images were collected and annotated by medical professionals.Here we build a convolutional neural network model that is able to classify the cells.


![193736032-b5847f1f-f002-4edc-912a-eaf48444f1b0](https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/468816d6-b927-4a2b-8212-e95abdfd6167)


## Neural Network Model:
<img width="608" alt="A" src="https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/0b9497b7-54a4-4d93-a401-6c082d76010f">


## DESIGN STEPS
### STEP 1:
Download and load the dataset to colab. After that mount the drive in your colab workspace to access the dataset.

### STEP 2:
Use ImageDataGenerator to augment the data and flow the data directly from the dataset directory to the model.

### STEP 3:
Split the data into train and test.

### STEP 4:
Build the convolutional neural network

### STEP 5:
Train the model with training data

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Plot the performance plot


## PROGRAM
```
Developed by: Vishranthi A
Registe number: 212221230124
```
```

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
from google.colab import drive
import tensorflow as tf
drive.mount('/content/drive')
!tar --skip-old-files -xvf '/content/drive/MyDrive/Dataset/cell_images.tar.xz' -C '/content/drive/MyDrive/Dataset/'
my_data_dir = '/content/drive/MyDrive/Dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[122]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[122])
plt.imshow(para_img)
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
model = models.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16

help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen
                             )
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
confusion_matrix(test_image_gen.classes,predictions)
plt.imshow(predictions)
print[predictions]
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![194067076-bf956f1e-9996-462e-97e7-f4427bc20d12](https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/fea52eca-8e64-400b-b1ba-7c25e0ea4b22)



### Classification Report

![194067099-5c583a7d-659f-42d5-a399-3de38d641070](https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/a30dc01f-1e81-477a-abe9-ea79a2f10b67)

### Confusion Matrix

![194067118-a5bb4f4d-d7af-4a55-817a-9ff8d16cd20d](https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/c329b2ac-38f9-45fd-9e02-4a3f8a8953c5)


### New Sample Data Prediction

![194067149-4ba8071f-c626-4673-ad06-4e6813ecfcd2](https://github.com/MEENA155/malaria-cell-recognition/assets/94677128/736f7e4d-525f-4a0b-8878-a4e3e56e2631)


## RESULT
Thus ,Successfully developed a convolutional deep neural network for Malaria Infected Cell Recognition.
