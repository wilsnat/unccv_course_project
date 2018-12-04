import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
import math
import glob
from matplotlib import pyplot as plt


def extractInput(image):
	# Taking a copy of the image
	img = image.copy()
	# Converting from BGR Colours Space to HSV
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Defining HSV Threadholds
	# lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
	# upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

	width = img.shape[0]
	height = img.shape[1] 

	#b,g,r = image.get(width/2, height/2)
	b = format(img[ math.floor(width/2), math.floor(height/2), 0])
	g = format(img[ math.floor(width/2), math.floor(height/2), 1])
	r = format(img[ math.floor(width/2), math.floor(height/2), 2])

	bgr_color = np.uint8([[[b, g, r]]])
	print('bgr--')
	print(bgr_color)
	hsvColor = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
	print('hsv--')
	print(hsvColor)

	lowerLimit = (hsvColor[0][0][0]-10,50, 90)
	upperLimit = (hsvColor[0][0][0]+10,255,255)
	
	lower_threshold = np.asarray([lowerLimit])
	upper_threshold = np.asarray([upperLimit])

	# Single Channel mask,denoting presence of colours in the about threshold
	imMask = cv2.inRange(img, lower_threshold, upper_threshold)

	# Cleaning up mask using Gaussian Filter
	imMask = cv2.GaussianBlur(imMask, (3, 3), 0)

	# Extracting im from the threshold mask
	im = cv2.bitwise_and(img, img, mask=imMask)

	# Return the Input image
	return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

	# Check for black
	hasBlack = False

	# Get the total number of occurance for each color
	occurance_counter = Counter(estimator_labels)

	# Quick lambda function to compare to lists
	def compare(x, y): return Counter(x) == Counter(y)

	# Loop through the most common occuring color
	for x in occurance_counter.most_common(len(estimator_cluster)):

		# Quick List comprehension to convert each of RBG Numbers to int
		color = [int(i) for i in estimator_cluster[x[0]].tolist()]

		# Check if the color is [0,0,0] that if it is black
		if compare(color, [0, 0, 0]) == True:
			# delete the occurance
			del occurance_counter[x[0]]
			# remove the cluster
			hasBlack = True
			estimator_cluster = np.delete(estimator_cluster, x[0], 0)
			break

	return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

	# Variable to keep count of the occurance of each color predicted
	occurance_counter = None

	# Output list variable to return
	colorInformation = []

	# Check for Black
	hasBlack = False

	# If a mask has be applied, remove th black
	if hasThresholding == True:

		(occurance, cluster, black) = removeBlack(
			estimator_labels, estimator_cluster)
		occurance_counter = occurance
		estimator_cluster = cluster
		hasBlack = black

	else:
		occurance_counter = Counter(estimator_labels)

	# Get the total sum of all the predicted occurances
	totalOccurance = sum(occurance_counter.values())

	# Loop through all the predicted colors
	for x in occurance_counter.most_common(len(estimator_cluster)):

		index = (int(x[0]))

		# Quick fix for index out of bound when there is no threshold
		index = (index-1) if ((hasThresholding & hasBlack)
							  & (int(index) != 0)) else index

		# Get the color number into a list
		color = estimator_cluster[index].tolist()

		# Get the percentage of each color
		color_percentage = (x[1]/totalOccurance)

		# make the dictionay of the information
		colorInfo = {"cluster_index": index, "color": color,
					 "color_percentage": color_percentage}

		# Add the dictionary to the list
		colorInformation.append(colorInfo)

	return colorInformation


def extractDominantColor(image, number_of_colors=3, hasThresholding=False):

	# Quick Fix Increase cluster counter to neglect the black(Read Article)
	if hasThresholding == True:
		number_of_colors += 1

	# Taking Copy of the image
	img = image.copy()

	# Convert Image into RGB Colours Space
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Reshape Image
	img = img.reshape((img.shape[0]*img.shape[1]), 3)

	# Initiate KMeans Object
	estimator = KMeans(n_clusters=number_of_colors, random_state=0)

	# Fit the image
	estimator.fit(img)

	# Get Colour Information
	colorInformation = getColorInformation(
		estimator.labels_, estimator.cluster_centers_, hasThresholding)
	return colorInformation


def plotColorBar(colorInformation):
	# Create a 500x100 black image
	color_bar = np.zeros((100, 500, 3), dtype="uint8")

	top_x = 0
	for x in colorInformation:
		bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

		color = tuple(map(int, (x['color'])))
		
		cv2.rectangle(color_bar, (int(top_x), 0),
					  (int(bottom_x), color_bar.shape[0]), color, -1)
		top_x = bottom_x
	return color_bar


"""## Section Two.4.2 : Putting it All together: Pretty Print

The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
	for x in color_info:
		print(pprint.pformat(x))
		print()


"""
The below lines of code, is the implementation of the above defined function.
"""

input_image_size = (187, 250)

data_path = 'dataset_full'
set_name = "set01"
#Pull in image filenames:
im_paths = glob.glob(data_path + '/' + set_name + '/*.jpg')
num_training_examples = 2
random_indices = np.arange(len(im_paths))
np.random.shuffle(random_indices)

training_indices = random_indices[:num_training_examples]

for count, index in enumerate(training_indices):
	#im = cv2.imread(im_paths[index])
	
	image = cv2.imread(im_paths[index])
	# Resize image to a width of 250
	image = imutils.resize(image, width=250)

	# Show image
	plt.subplot(3, 1, 1)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title("Original Image")
	# plt.show()

	# Apply Input Mask
	im = extractInput(image)

	plt.subplot(3, 1, 2)
	plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
	plt.title("Thresholded  Image")
	# plt.show()

	# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
	dominantColors = extractDominantColor(im, hasThresholding=True)

	# Show in the dominant color information
	#print("Color Information")
	#prety_print_data(dominantColors)

	# Show in the dominant color as bar
	print("Color Bar")
	print(dominantColors)
	colour_bar = plotColorBar(dominantColors)
	plt.subplot(3, 1, 3)
	plt.axis("off")
	plt.imshow(colour_bar)
	plt.title("Color Bar")

	plt.tight_layout()
	plt.show()


