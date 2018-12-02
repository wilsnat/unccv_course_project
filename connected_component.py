import sys, os
import numpy as np
import cv2
'''import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from easydict import EasyDict'''

import matplotlib.pyplot as plt
from data_loader import data_loader

img = cv2.imread('dataset_full/resized_img/0018.jpg')
#cv2.imshow('test img',img)
#cv2.waitKey(0)

mask = np.zeros(img.shape[:2], np.uint8)

bgMask = np.zeros((1,65),np.float64)
fgMask = np.zeros((1,65),np.float64)

rect = (50,50,200,200)

cv2.grabCut(img, mask, rect, bgMask, fgMask, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow('test img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 10, 250)
#(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_, cnts, _= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

idx = 0
for c in cnts:
	x,y,w,h = cv2.boundingRect(c)
	if w>50 and h>50:
		idx+=1
		new_img=img[y:y+h,x:x+w]
		#cv2.imwrite(str(idx) + '.png', new_img)
cv2.imshow("im",new_img)
cv2.waitKey(0)'''