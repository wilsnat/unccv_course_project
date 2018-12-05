import numpy as np
import math
from easydict import EasyDict
import glob
import cv2
import matplotlib.pyplot as plt
import pdb;

hls_sucks = False

def data_vis(train_test_split = 1,
				input_image_size = (187, 250),
				data_path = 'dataset_full',set = "set01", classification_offset = 0):
	exifpath = open(data_path + '/exif' + set[-2:] + '.csv', 'r', encoding = 'utf-8-sig')
	exif = np.genfromtxt(exifpath, delimiter=',', dtype='U20')
	exifpath.close
	exif = exif[1:,0:4:3]
	exif[:,0] = [x.split("/")[1] for x in exif[:,0]]
	exif[:,1] = exif[:,1].astype(int)

	labpath = open(data_path + '/lab_colors' + set[-2:] + '.csv', 'r', encoding='utf-8-sig')
	lab = np.genfromtxt(labpath, delimiter=',', dtype='float32')
	lab_samples_per_color = lab[0,0]
	lab = lab[1:]
	labpath.close
	bgr = cv2.cvtColor(np.asarray([lab]), cv2.COLOR_Lab2BGR)
	#bgr = bgr.squeeze()
	if hls_sucks:
		ycolor = bgr.squeeze()
	else:
		hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
		ycolor = hls.squeeze()

	#Pull in image filenames:
	im_paths = glob.glob(data_path + '/' + set + '/*.jpg')

	#Train test split
	num_training_examples = int(np.round(train_test_split*len(im_paths)))
	num_testing_examples = len(im_paths) - num_training_examples

	indices = np.arange(len(im_paths))

	#Make easydicts for data
	data = EasyDict()
	data.train = EasyDict()
	data.test = EasyDict()

	# Make empty arrays to hold data:
	data.train.im = np.zeros((num_training_examples, 3),
							dtype = 'float32')
	data.train.ex = np.zeros((num_training_examples, 2), dtype = 'float32')
	data.train.y = np.zeros((num_training_examples, 3), dtype = 'float32')
	data.train.yi = np.zeros((num_training_examples, 1), dtype = int)

	data.test.im = np.zeros((num_testing_examples, input_image_size[0], input_image_size[1], 3),
							dtype = 'float32')
	data.test.ex = np.zeros((num_testing_examples, 2), dtype = 'float32')
	data.test.y = np.zeros((num_testing_examples, 3), dtype = 'float32')
	data.test.yi = np.zeros((num_testing_examples, 1), dtype = int)

	for count, index in enumerate(indices):
		imin = np.float32(cv2.imread(im_paths[index]))/255
		if not hls_sucks:
			imin = cv2.cvtColor(imin, cv2.COLOR_BGR2HLS)
		if(imin.shape[0] < imin.shape[1]):
			data.train.im[count] = imin[int(input_image_size[0]/2), int(input_image_size[1]/2)]
		else:
			imin = imin.swapaxes(0,1)
			data.train.im[count] = imin[int(input_image_size[0]/2), int(input_image_size[1]/2)]
		data.train.ex[count] = exif[index]
		data.train.yi[count] = math.floor(index/lab_samples_per_color)+classification_offset
		data.train.y[count] = ycolor[math.floor(index/lab_samples_per_color)]

	#graph here

	#chart here
	num_visible = 32
	plt.title("input colors")
	pdb.set_trace()
	if not hls_sucks:
		data.train.y =  cv2.cvtColor(np.asarray([data.train.y]), cv2.COLOR_HLS2RGB).squeeze()
		data.train.im =  cv2.cvtColor(np.asarray([data.train.im]), cv2.COLOR_HLS2RGB).squeeze()
	else:
		data.train.y[0], data.train.y[2] = data.train.y[2], data.train.y[0]
		data.train.im[0], data.train.im[2] = data.train.im[2], data.train.im[0]
	palette = np.concatenate((data.train.y[::10],data.train.im),axis=0)
	first_value = np.arange(0,num_visible).reshape(num_visible,1)
	input_colors = np.arange(32,num_visible*10+32).reshape(-1,10)
	picture = np.concatenate((input_colors,first_value),axis=1)
	plt.imshow(palette[picture.T])
	plt.show()

	return data

data_vis()
