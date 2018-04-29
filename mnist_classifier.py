##########################################
# Copyright (c) Javier Cabero Guerra 2018
# Licensed under MIT
##########################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

import numpy as np

class MNISTClassifier:

	__model_file = 'keras_mnist_cnn_topology.json'
	__weights_file = 'keras_mnist_cnn_weights.h5'
	__model = None

	__digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	def __init__(self):
		""" Creates and trains a MNIST dataset classifier """
		self.load_cnn(self.__model_file, self.__weights_file)

	def load_cnn(self, model_file, weights_file):
		model = self.load_cnn_topology(model_file)
		print('Loading weights...')
		model.load_weights(weights_file)
		print("Compiling...")
		model = self.compile(model)
		print("Compiled!")
		print('Model' ,model)
		self.__model = model

	def load_cnn_topology(self, model_file):
		""" Loads the neural network architecture from a file """
		json_file = open(model_file)
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		print("Loaded cnn topology from disk")
		return model

	def compile(self, model): 
		model.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])
		return model

	def prediction_percentages_to_digit_label(self, prediction_percentages):
	    max_idx = np.argmax(prediction_percentages)
	    return self.__digit_labels[max_idx]

	def predict(self, sample):
		sample_array = np.asarray([sample])
		print(self.__model)
		print("Predicting...")
		prediction_percentages = self.__model.predict(sample_array)
		print("Predicted!")
		digit_label = self.prediction_percentages_to_digit_label(prediction_percentages)
		return digit_label