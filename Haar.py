#################################################
#
# Authors: James Hynes, Gytis Mackevicius, Aaron Renaghan, Seamus Timmons 
#
# Project Title: Finding Regions of Face in Images
#
# Introduction: This program is designed to locate faces within images by using Image Processing technuiqes and Machine Learning
#
# Initial Thoughts and Ideas: James  - 
#							  Gytis  - 
#							  Aaron  - I like the sounds of this project, looks quite complex perhaps we can threshold for skin initially
#									   and then from there all we should have to do is distinguish between skin and faces, maybe holes left
#									   in the faces such as mouths will help us identify the differance? 
#							  Seamus - 
#
# Final Thoughts and Reflections: James  -
#			 					  Gytis  -
#			 					  Aaron  -
#			 					  Seamus -
#
# Start Date: September Somthingth
#
# Finish Date: 16/11/2017
#
# Algorithm Description: The Algorithm is a rather simple one, here it is broken down step by step
#	Step 1. Import Libraries For use in the code
#   Step 2. Load Our Classifier, We will need further down
#   Step 3. User Selects their image and we extract the images dimensions
#   Step 4. We Resize our image before making it GreyScale, This helps with our Haar Cascades as it will not work on (Low Pixel count faces? -- Someone reword that) 
#   Step 5. Run the grayscale image through the classifier search all possible boxes that could contain a face and then draw a rectangle where there is a match
#   Step 6. Resize our image back to its origional state
#   Step 7. Display and write final result to file


# Importing our Libraries

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

# The Classifier we used provided by OpenCV
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# Reading in our Image and then get its dimensions
originalImage = cv2.imread(easygui.fileopenbox(msg = 'Select Image For Face Detection'))
originalHeight, originalWidth, bpp = np.shape(originalImage)

# Resizing Our Image and Turning it to GrayScale to improve accuracy of our Haar Cascades later on
hugeImage = cv2.resize(originalImage,None,fx=5, fy=5, interpolation = cv2.INTER_CUBIC) #Upscaling the image to accomodate Haar's
grayscale = cv2.cvtColor(hugeImage, cv2.COLOR_BGR2GRAY)
	
# Running the Haar Cascades and drawing boxes around detected faces
faceDetection = face_cascade.detectMultiScale(grayscale, 1.3, 5)
for (x,y,w,h) in faceDetection:
	cv2.rectangle(hugeImage,(x,y),(x+w,y+h),(255,0,0),5)
	
# Resizing the image back to its origional size 
finalResult = cv2.resize(hugeImage, (originalWidth, originalHeight)) #Resizing the image to fit the screen

# Writing the image to file and displaying it
cv2.imwrite('FacesDetected.png', finalResult)
cv2.imshow('Detected Faces', finalResult)
cv2.waitKey(0)
cv2.destroyAllWindows()
