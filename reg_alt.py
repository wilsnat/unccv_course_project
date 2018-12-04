import sys, os
import numpy as np
import math
import tensorflow as tf
#import tensorlayer as tl
from tensorflow import keras
from datetime import datetime
from easydict import EasyDict
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.merge import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
#import pdb;
from data_loader import data_loader

def main():
	#Make easydicts for data
	train_test_split = 0.8

	data = EasyDict()
	data.train = EasyDict()
	data.test = EasyDict()
	
	input_image_size = (187, 250)

	data_path = 'dataset_full'
	set_name = "set01"
	#Pull in image filenames:
	im_paths = glob.glob(data_path + '/' + set_name + '/*.jpg')

	labpath = open(data_path + '/train_y.csv' , 'r', encoding='utf-8-sig')
	lab = np.genfromtxt(labpath, delimiter=',', dtype='float32')

	num_training_examples = int(np.round(train_test_split*len(im_paths)))
	num_testing_examples = len(im_paths) - num_training_examples

	random_indices = np.arange(len(im_paths))
	np.random.shuffle(random_indices)

	training_indices = random_indices[:num_training_examples]
	testing_indices = random_indices[num_training_examples:]

	lab = lab[:len(lab),1:4]
	bgr = cv2.cvtColor(np.asarray([lab]), cv2.COLOR_Lab2BGR)
	hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	train_y = hls.squeeze()

	#Make easydicts for data
	data = EasyDict()
	data.train = EasyDict()
	data.test = EasyDict()

	# Make empty arrays to hold data:
	data.train.im = np.zeros((num_training_examples, input_image_size[0], input_image_size[1], 3), dtype = 'float32')
	data.train.y = np.zeros((num_training_examples, 3), dtype = 'float32')
	data.test.im = np.zeros((num_testing_examples, input_image_size[0], input_image_size[1], 3), dtype = 'float32')
	data.test.y = np.zeros((num_testing_examples, 3), dtype = 'float32')

	for count, index in enumerate(training_indices):
		im = cv2.imread(im_paths[index])
		
		im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) 

		if(im.shape[0] < im.shape[1]):
			data.train.im[count, :, :, :] = im
		else:
			data.train.im[count, :, :, :] = im.swapaxes(0,1)

		data.train.y[count] = train_y[index]

	for count, index in enumerate(testing_indices):
		im = cv2.imread(im_paths[index])
		im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) 

		if(im.shape[0] < im.shape[1]):
			data.test.im[count, :, :, :] = im
		else:
			data.test.im[count, :, :, :] = im.swapaxes(0,1)

		data.test.y[count] = train_y[index]
	#print(data.train.im)

	print('--------------------------')
	print(data.train.im.shape)
	print('--------------------------')
	print(data.train.y.shape)
	

	#all_data = np.concatenate((data.train.im, data.train.y))

	# all_data = EasyDict()
	# all_data.train.im = data.train.im
	# all_data.train.y = data.train.y
	# all_data.test.im = data.test.im
	# all_data.test.y = data.test.y
	
	model = build_model(data)
	model.summary()

	EPOCHS = 5000

	# Store training stats
	history = model.fit(data.train.im, data.train.y, batch_size=16,
						epochs=EPOCHS,
						verbose=0,
						callbacks=[PrintDot()])
	plot_history(history)

	data_predictions = predict(model,data.test)

	print("")
	print(data_predictions)
	print("-----------------------------")
	#print("set01 center pixel test mae: " + str(np.abs(np.subtract(data_out.test.y,data_out.test.full[:,-3:])).mean()))
	#print("set01 model test mae: " + str(np.abs(np.subtract(data_out.test.y.flatten(), data_predictions)).mean()))


	
def build_model(data):
	model = keras.Sequential([
		keras.layers.Dense(64, activation=tf.nn.relu,
						   input_shape=(data.train.im.shape[1],)),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(3)
		])
	optimizer = tf.train.RMSPropOptimizer(0.0002)

	model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mae'])
	return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		if epoch % 300 == 0: print(str(epoch) + "th epoch and the loss is " + str(logs['loss']))
		print('.', end='')

	#plot history of the training run (we could implement tensorboard later)
	def plot_history(history):
		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Abs Error [1000$]')
		plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
		plt.plot(history.epoch, np.array(history.history['loss']),label = 'Val loss')
		plt.legend()
		plt.show()

#predict and show plot
def predict(model, data):
	test_predictions = model.predict(data.test.im).flatten()
	return test_predictions

#plot x,y. to show plot make sure to .show() the returned plot
#extra note: check to see if your model predictions are better than
# plot(data.test.im.center_pixel, data.test.y)
def plot(x,y):
	plt.title('data')
	plt.scatter(x, y)
	plt.xlabel('True Values')
	plt.ylabel('Predictions')
	plt.axis('equal')
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())
	_ = plt.plot([-100, 100], [-100, 100])
	return plt

def print_colors(x,y,z,title="color comparison"):
	plt.title(title)
	if hls_sucks:
		rgbx = (x*255).astype('int')
		rgby = (y*255).astype('int')
		rgbz = (z*255).astype('int')
	else:
		x[:,0] = x[:,0]*360
		y[:,0] = y[:,0]*360
		z[:,0] = z[:,0]*360
		rgbx = (cv2.cvtColor(np.asarray([x]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
		rgby = (cv2.cvtColor(np.asarray([y]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
		rgbz = (cv2.cvtColor(np.asarray([z]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
	print(np.max(rgbx))
	palette = np.concatenate((rgbx,rgby,rgbz),axis=0)
	first_value = np.arange(0,rgbx.shape[0]-1)
	second_value = np.arange(rgbx.shape[0],2*rgbx.shape[0]-1)
	y_value = np.arange(2*rgbx.shape[0],3*rgbx.shape[0]-1)
	picture = np.concatenate(([y_value],[first_value],[second_value],[y_value]),axis=0)
	#this prints backwards on the graph, its actually y,yhat,pixel_data,y
	plt.ylabel('color([y],[pixel data],[yhat],[y])')
	plt.imshow(palette[picture])
	plt.show()
main();