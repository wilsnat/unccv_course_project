import numpy as np
from colorsys import rgb_to_hsv
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import math

colors = dict((
			((196, 2, 51), "RED"),
			((255, 165, 0), "ORANGE"),
			((255, 205, 0), "YELLOW"),
			((0, 128, 0), "GREEN"),
			((0, 0, 255), "BLUE"),
			((127, 0, 255), "VIOLET"),
			((0, 0, 0), "BLACK"),
			((255, 255, 255), "WHITE"),))

input_image_size = (187, 250)

data_path = 'dataset_full'
set_name = "set01"
#Pull in image filenames:
im_paths = glob.glob(data_path + '/' + set_name + '/*.jpg')
num_training_examples = 10
random_indices = np.arange(len(im_paths))
np.random.shuffle(random_indices)

training_indices = random_indices[:num_training_examples]



bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#rect = (50,50,250,250)
rect = (25,25,225,225)


for count, index in enumerate(training_indices):
	im = cv2.imread(im_paths[index])
	
	mask = np.zeros(im.shape[:2], np.uint8)
	cv2.grabCut(im, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
	im_array = im*mask2[:,:,np.newaxis]
	cv2.imshow('test img',im_array)
	cv2.waitKey(0)
	
	im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB) 
	
	avg_color_per_row = np.average(im, axis=0)
	avg_color = np.average(avg_color_per_row, axis=0)

	#print(avg_color)
	print(index)
	print(min((abs(rgb_to_hsv(*k)[0]-rgb_to_hsv(*avg_color)[0]),v) for k,v in colors.items()))
	print("-------------------")



	