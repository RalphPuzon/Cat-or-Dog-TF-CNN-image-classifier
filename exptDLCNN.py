#Convolutional Neural Networks:

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

os.chdir('C:\\Users\\Ralph\\Desktop\\Courses\\DeepLearning\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)')

#PREPROCESSING: the data is preprocessed via splitting up the test and
#training folders

#BUILDING THE CNN:
from keras.models import Sequential    # Artificial Neural Network
from keras.layers import Convolution2D # Convolution Step
from keras.layers import MaxPooling2D  # Pooling Step
from keras.layers import Flatten       # Flattening of grid for ANN
from keras.layers import Dense         # Artificial Neural Network
from keras.layers import Dropout       # Dropout
from keras import optimizers           # Optimizers

#CNN INITIALIZATION:
classifier = Sequential()
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), 
                             input_shape = (32, 32, 3),
                             activation = 'relu',
                             padding='same'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten()) #no params, keras understands to use previous layer via classifier.add format

#Artificial Neural Network layer (FULL CONNECTION):
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#note on output layer:
# input_dim = 1, since we're using a probability, 0 is cat and 1 is dog
#although not really sure of order here

#Manual creation of the optimizer for lr tuning:
Rpro = optimizers.RMSprop(lr = 0.0006)

#COMPILING THE CNN:
classifier.compile(optimizer = Rpro, loss = 'binary_crossentropy', metrics = ['accuracy'])

#FITTING THE IMAGES INTO THE CNN:
#image augmentation: reduce overfitting without needing a large sample size
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory( 
        'dataset/training_set',
        target_size= (32,32),
        batch_size=32,
        class_mode='binary')

test_set = train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size= (32,32),
        batch_size= 32,
        class_mode= 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 250,
                         validation_data=test_set,
                         validation_steps = (2000/32))
 
#single image prediction:
#import numpy as np -- already imported above


from skimage.io import imread
from skimage.transform import resize

class_labels = {v:  k for k, v in training_set.class_indices.items()}

pred_img = imread('dataset/single_prediction/cat_or_dog_1.jpg')
pred_img = resize(pred_img,(32,32)) #resize size must be same as training size
pred_img = np.expand_dims(pred_img, axis = 0) 
if (np.max(pred_img) >1):
    pred_img = pred_img/255.0

result = classifier.predict_classes(pred_img)
print(class_labels[result[0][0]])


class_labels = {v: k for k, v in training_set.class_indices.items()} 

pred_img2 = imread('dataset/single_prediction/cat_or_dog_2.jpg')
pred_img2 = resize(pred_img2,(32,32))
#resize size must be same as training size
pred_img2 = np.expand_dims(pred_img2, axis = 0) 
if (np.max(pred_img2) >1):
    pred_img = pred_img2/255.0

result2 = classifier.predict_classes(pred_img2)
print(class_labels[result2[0][0]])

