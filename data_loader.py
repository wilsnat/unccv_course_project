import numpy as np
import math
from easydict import EasyDict
import glob
import cv2
import pdb;

def data_loader(train_test_split = 0.7,
				input_image_size = (187, 250),
				data_path = 'dataset_full',set = "set01"):
	#,exif_include = True
	#if(exif_include):
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
	hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
	hls = hls.squeeze()

	#Pull in image filenames:
	im_paths = glob.glob(data_path + '/' + set + '/*.jpg')

	#Train test split
	num_training_examples = int(np.round(train_test_split*len(im_paths)))
	num_testing_examples = len(im_paths) - num_training_examples

	random_indices = np.arange(len(im_paths))
	np.random.shuffle(random_indices)

	training_indices = random_indices[:num_training_examples]
	testing_indices = random_indices[num_training_examples:]

	#Make easydicts for data
	data = EasyDict()
	data.train = EasyDict()
	data.test = EasyDict()

	# Make empty arrays to hold data:
	data.train.im = np.zeros((num_training_examples, input_image_size[0], input_image_size[1], 3),
							dtype = 'float32')
	data.train.ex = np.zeros((num_training_examples, 2), dtype = 'float32')
	data.train.y = np.zeros((num_training_examples, 3), dtype = 'float32')

	data.test.im = np.zeros((num_testing_examples, input_image_size[0], input_image_size[1], 3),
							dtype = 'float32')
	data.test.ex = np.zeros((num_testing_examples, 2), dtype = 'float32')
	data.test.y = np.zeros((num_testing_examples, 3), dtype = 'float32')

	for count, index in enumerate(training_indices):
		imin = np.float32(cv2.imread(im_paths[index]))/255
		im = cv2.cvtColor(imin, cv2.COLOR_BGR2HLS)

		if(im.shape[0] < im.shape[1]):
			data.train.im[count, :, :, :] = im
		else:
			data.train.im[count, :, :, :] = im.swapaxes(0,1)
		data.train.ex[count] = exif[index]
		data.train.y[count] = hls[math.floor(index/lab_samples_per_color)]

		'''print(im_paths[index])
		print(data.train.y[count])
		print('------')'''

	for count, index in enumerate(testing_indices):
		imin =  np.float32(cv2.imread(im_paths[index]))/255
		im = cv2.cvtColor(imin, cv2.COLOR_BGR2HLS)
		if(im.shape[0] < im.shape[1]):
			data.test.im[count, :, :, :] = im
		else:
			data.test.im[count, :, :, :] = im.swapaxes(0,1)
		data.test.ex[count] = exif[index]
		data.test.y[count] = hls[math.floor(index/lab_samples_per_color)]

	print('Loaded', str(len(training_indices)), 'training examples and ', str(len(testing_indices)), 'testing examples. ')
	print('data.*.im = image in hls')
	print('data.*.ex = exif data')
	print('data.*.y = out in hls')

	return data
