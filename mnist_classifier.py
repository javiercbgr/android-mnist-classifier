#########################################################################
# Copyright (c) Javier Cabero Guerra 2018
# Licensed under MIT
#
# Neural network architecture extracted from: 
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py 
# 
# Dependencies: 
#   python-mnist - https://pypi.org/project/python-mnist/
#########################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

from mnist import MNIST
mndata = MNIST('./mnist-dataset')
mndata.gz = True
images, labels = mndata.load_training()

# Convert to numpy arrays
images = np.asarray(images)
labels = np.asarray(labels)


def create_training_test_sets(images, labels, training_split_percentage):
    """
        training_split_percentage is the percentage of the data that
                                  will go into training
    """
    train_sample_count = np.floor(training_split_percentage * len(images)).astype(int)
    test_sample_count = len(images) - train_sample_count
    idxs_array = np.arange(0, len(images), 1)
    idxs_train = np.random.choice(idxs_array, train_sample_count, replace=False)
    idxs_test  = [x for x in idxs_array if x not in idxs_train]
    x_train = images[idxs_train]
    y_train = labels[idxs_train]
    x_test = images[idxs_test]
    y_test = labels[idxs_test]
    return (x_train, y_train), (x_test, y_test)
    
(x_train, y_train), (x_test, y_test) = create_training_test_sets(images, labels, 0.8)

# Reshape data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normaliza data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])