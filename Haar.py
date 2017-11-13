#################################################
#
# Authors: James Hynes, Gytis Mackevicius, Aaron Renaghan, Seamus Timmons 
#
# Project Title: Finding Regions of Face in Images
#
# Introduction: This program is designed to locate faces within images by using Image Processing technuiqes and Machine Learning
#
# Initial Thoughts and Ideas: James  - 
#                             Gytis  -
#                             Aaron  - I like the sounds of this project, looks quite complex perhaps we can threshold
#                                      for skin initially and then from there all we should have to do is distinguish
#                                      between skin and faces, maybe holes left in the faces such as mouths will help us
#                                      identify the differance?
#                             Seamus - My initial thoughts on the project are that it will be quite difficult. While
#                                      faces seem very distinct I imagine it wont be very easy to classify them for a
#                                      computer. We have learned about thresholding skin which could help, but I feel it
#                                      will be a challenging project.
#
# Final Thoughts and Reflections: James  -
#                                 Gytis  -
#                                 Aaron  -
#                                 Seamus - My final thoughts on the project are sunrising. While I initially though it
#                                          would be very difficult our goal, it turned out to be a lot easier. Haar
#                                          cascades turned out to be the simplest way to do it. OpenCV provides trained
#                                          models for Haar so we don't have to train it (which is outside the scope of
#                                          the project). I found the project very interesting and enjoyable overall, and
#                                          thanks to Aaron who took the lead on the project giving us direction.
#
#
# Start Date: September Somthingth
#
# Finish Date: 16/11/2017
#
# Algorithm Description: The Algorithm is a rather simple one, here it is broken down step by step
#    Step 1. Import Libraries For use in the code
#   Step 2. Load Our Classifier, We will need further down
#   Step 3. User Selects their image and we extract the images dimensions
#   Step 4. We Resize our image before making it GreyScale, This helps with our Haar Cascades as it will not work on
#           (Low Pixel count faces? -- Someone reword that)
#   Step 5. Run the grayscale image through the classifier search all possible boxes that could contain a face
#   Step 6. Check for a percentage of skin pixels using thresholding
#   Step 7. Draw box around each with that percentage
#   Step 8. Resize our image back to its origional state
#   Step 9. Display and write final result to file


# Importing our Libraries

import sys
import logging as log
import numpy as np
import cv2
import easygui


# Method to load a cascade classifier
def loadCascadeClassifier(fileName):

    log.info("Loading Cascade Classifier")
    log.debug("Loading file \"%s\"" % fileName)

    try:
        cascade = cv2.CascadeClassifier(fileName)
    except Exception as e:
        template = "An exception of type {0} occurred with arguments:\n {1!r}"
        message = template.format(type(e).__name__, e.args)
        log.critical(message)
        sys.exit()

    return cascade


# Method to load in an image
def loadImage(fileName):

    log.info("Loading image")
    log.debug("Loading file \"%s\"" % fileName)

    try:
        image = cv2.imread(fileName)
    except Exception as e:
        template = "An exception of type {0} occurred with arguments:\n {1!r}"
        message = template.format(type(e).__name__, e.args)
        log.critical(message)
        sys.exit()

    return image


# Method to check if detected region has skin in it
def detectSkin(image):

    log.info("Detecting for skin")
    log.info("Getting image properties")

    height, width, channels = image.shape

    log.info("Converting image to YUV color space")

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    log.info("Extracting out U and V channels")

    u = yuv[:,:,1]
    v = yuv[:,:,2]

    log.info("Masking out U and V channels for skin")

    uMask = cv2.inRange(u, 105, 135)
    vMask = cv2.inRange(v, 140, 165)

    mask = cv2.bitwise_not(cv2.bitwise_and(uMask, vMask))

    blackCount = 0
    totalPixels = height * width

    log.info("Getting skin pixels")

    for i in range(1, width):
        for j in range(1, height):
            if mask[j,i] == 0:
                blackCount += 1

    log.info("Getting skin percentage")

    percentSkin = round((blackCount * 100) / totalPixels)

    log.info("Checking skin percentage")

    if percentSkin >= 10:
        return True

    return False

# Method to perform Haar Cascades on an image
def haarCascades(image, classifier):

    log.info("Performing face recognition")
    log.debug("Getting image properties")

    # Getting image dimensions
    originalHeight, originalWidth, bpp = np.shape(image)

    log.info("Upscaling image")

    # Upscaling the image to accomodate Haar's
    hugeImage = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    log.info("Converting image to greyscale")

    # Converting the image to greyscale to accomodate Haar's
    grayscale = cv2.cvtColor(hugeImage, cv2.COLOR_BGR2GRAY)

    log.info("Performing Haar cascade detection")

    # Running the Haar Cascades and drawing boxes around detected faces
    faceDetection = classifier.detectMultiScale(grayscale, 1.3, 5)

    log.info("Drawing boxes around faces")

    # Drawing boxes around detected regions of face
    for (x, y, w, h) in faceDetection:
        if detectSkin(hugeImage[y:y+h, x:x+w]):
            cv2.rectangle(hugeImage, (x, y), (x + w, y + h), (255, 0, 0), 5)

    log.info("Downscaling image to original size")

    # Resizing the image back to its origional size
    result = cv2.resize(hugeImage, (originalWidth, originalHeight))  # Resizing the image to fit the screen

    return result


# Setting log file location and level
log.basicConfig(filename='haar.log', level=log.INFO)

# The Classifier we used provided by OpenCV
face_classifier = loadCascadeClassifier('haarcascade_frontalface_default.xml')

# Reading in our Image and then get its dimensions
originalImage = loadImage(easygui.fileopenbox(msg='Select Image For Face Detection'))

# Performing face detection on Haar cascades
finalResult = haarCascades(originalImage, face_classifier)

# Writing the image to file and displaying it
cv2.imwrite('FacesDetected.png', finalResult)
cv2.imshow('Detected Faces', finalResult)

cv2.waitKey(0)
cv2.destroyAllWindows()