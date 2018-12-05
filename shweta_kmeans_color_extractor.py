import cv2
import numpy as np 
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
import math
import glob
import webcolors
from matplotlib import pyplot as plt
from matplotlib import colors as clr


def main():
    input_image_size = (187, 250)

    data_path = 'dataset_full'
    set_name = "set01"
    #Pull in image filenames:
    im_paths = glob.glob(data_path + '/' + set_name + '/*.jpg')
    num_training_examples = 10
    random_indices = np.arange(len(im_paths))
    np.random.shuffle(random_indices)
    #randomly select num_training_examples images
    training_indices = random_indices[:num_training_examples]

    for count, index in enumerate(training_indices):
        #read image data
        image = cv2.imread(im_paths[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img = image
        mask = np.zeros(img.shape[:2], np.uint8)

        bgMask = np.zeros((1,65),np.float64)
        fgMask = np.zeros((1,65),np.float64)

        width,height,channel = image.shape
        rect = (round(0.2*width),round(0.2*height),round(0.8*width),round(0.8*height))

        #foreground object detection
        cv2.grabCut(img, mask, rect, bgMask, fgMask, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]

        number_of_colors = 5

        #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv_img = img

        img = img.reshape((img.shape[0]*img.shape[1]), 3)

        # Initiate KMeans Object
        estimator = KMeans(n_clusters=number_of_colors, random_state=0)

        # Fit the image
        estimator.fit(img)

        # Get Colour Information
        hasThresholding = True
        colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)

        # Show image
        plt.subplot(3, 1, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(3, 1, 2)
        plt.imshow(hsv_img)
        plt.title("Foreground Image")

        colour_bar = plotColorBar(colorInformation)
        color_name = getDominantColorName(colorInformation)
        plt.subplot(3, 1, 3)
        plt.axis("off")
        plt.imshow(colour_bar)
        plt.title("Color Bar")
        plt.title("Color Name - "+color_name)

        plt.tight_layout()
        plt.show()


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
        index = (index-1) if ((hasThresholding & hasBlack) & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color, "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation

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

def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))
        
        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar

def getDominantColorName(colorInformation):
    
    clrPercent = []
    for x in colorInformation:
        clrPercent.append(x["color_percentage"])

    max_percent_color = np.max(np.asarray(clrPercent))
    for x in colorInformation:
        if (max_percent_color == x["color_percentage"]):
            color = tuple(map(int, (x['color'])))
    
    hsv_color = np.uint8([[[color[2], color[1], color[0]]]])
    bgr_color = hsv_color.squeeze()
    
    requested_colour = (bgr_color[2], bgr_color[1], bgr_color[0])
    color_code = ",".join(map(str, requested_colour)) 

    #fetch color name using (r,g,b value)
    actual_name, closest_name = get_colour_name(requested_colour)

    return closest_name+'('+color_code+')';    

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name



def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

main()