# Haars cascades

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
img = cv2.imread('graduation.jpg')

# get image dimensions for width, height and channels
height,width, bpp = np.shape(img)

img = cv2.resize(img,None,fx=5, fy=5, interpolation = cv2.INTER_CUBIC) #Upscaling the image to accomodate Haar's
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)

# Resizing image to original size
small = cv2.resize(img,(width,height)) 

cv2.imshow('img',small)
cv2.waitKey(0)
cv2.destroyAllWindows()