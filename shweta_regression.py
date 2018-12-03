import sys, os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow import keras
from datetime import datetime
from easydict import EasyDict
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pdb
#import pdb;
from data_loader import data_loader

dMode = "metrics_mode"

def main():
	data_out = import_and_prep_datasets(i_train_test_split = 0.9,  p_mode = dMode)
	#print(data_out.train.y)

	sess = tf.InteractiveSession()
	place_x = tf.placeholder(tf.float32, [None, data_out.train.im.shape[1], 
										data_out.train.im.shape[2], 
										3])
	place_y = tf.placeholder(tf.float32, [None, 3])

	'''place_x = tf.placeholder(tf.float32, [None, data_out.train.im.shape[0]*data_out.train.im.shape[1]*data_out.train.im.shape[2]*data_out.train.im.shape[3]])
	place_y = tf.placeholder(tf.float32, [None, data_out.train.y.shape[0]*data_out.train.im.shape[1]])

	print(data_out.train.y.shape)

	training_x = data_out.train.im.flatten()
	training_x = np.expand_dims(training_x, axis=0)

	print(training_x.shape)

	training_y = data_out.train.y.flatten()
	training_y = np.expand_dims(training_y, axis=0)
	print(training_y.shape)'''

	'''training_x = data_out.train.im
	training_y = data_out.train.y

	network_1 = tl.layers.InputLayer(place_x, name='input_layer_1')
	flattened_input = tf.reshape(network_1, [-1, 6*6*256])
	hidden = tl.layers.DenseLayer(flattened_input, n_units = 3, act = tf.nn.elu, name='hidden')
	network_2 = tl.layers.DenseLayer(hidden, n_units = 1, act = tf.identity, name='output_2')
	hidden_output = hidden.outputs
	final_output = network_2.outputs
	cost = tl.cost.mean_squared_error(final_output, place_y)
	optimize = tf.train.AdamOptimizer().minimize(cost)
	training_epochs = 10000
	# Train model 
	sess.run(tf.global_variables_initializer())
	for i in range(training_epochs):
		feed_dict = {
			place_x: training_x, 
			place_y: training_y
		}
	_cost, _hid_out, _final_out, _ = sess.run([cost, hidden_output, final_output, optimize], feed_dict=feed_dict)

	plt.plot(range(len(_cost)),_cost)
	plt.axis([0,training_epochs,0,np.max(_cost)])
	plt.show()

	print("train of shape: " + str(data_out.train.im.shape) + " train y of shape: " + str(data_out.train.y.shape))'''
	#print("train of shape: " + str(data_out.train.im.shape) + " test of shape: " + str(data_out.test.im.shape))

#intermediary function. you can just use data_loader() then data_prep() if you'd like
def import_and_prep_datasets(i_train_test_split = 0.7,
				i_input_image_size = (187, 250),
				i_data_path = 'dataset_full',
				i_set = "set01", p_mode = "center_mode",
				p_slice_mode_slice = 30):
	return data_loader(train_test_split=i_train_test_split,
		input_image_size=i_input_image_size,
		data_path = i_data_path,
		set=i_set)

	#return data_prep(data, mode=p_mode, slice_mode_slice=p_slice_mode_slice)

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

main()

'''training_data = []
y_labels = np.genfromtxt('dataset_full/train_y.csv', delimiter = ',')
train_y = []
for i in range(y_labels.shape[0]):
	#print(i)
	train_y.append(y_labels[i][2:])
	
print(train_y)	

img_path = 'dataset_full/resized_img'
for im in os.listdir(img_path):
	img = cv2.imread(os.path.join(img_path,im))
	
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(gray, 10, 250)
	#(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	_, cnts, _= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	idx = 0
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		if w>50 and h>50:
			idx+=1
			im_array=img[y:y+h,x:x+w]

			#x = im_array.flatten()
			#x = np.expand_dims(x, axis=0)			

train_x = im_array.flatten()
train_x = np.expand_dims(train_x, axis=0)


#print(im_array)
#cv2.imshow('test img',im_array[100:])
#cv2.waitKey(0)
'''