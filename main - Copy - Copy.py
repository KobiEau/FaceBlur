# import necessary libraries
import cv2
import numpy as np


# function to apply Gaussian blur to an image
def blur(img, k):
    # get the height and width of the image
    h, w = img.shape[:2]
    # calculate the size of the Gaussian kernel based on the block size
    kh, kw = h // k, w // k
    # make sure the kernel size is odd
    if kh % 2 == 0:
        kh -= 1
    if kw % 2 == 0:
        kw -= 1
    # apply Gaussian blur to the image
    img = cv2.GaussianBlur(img, ksize=(kh, kw), sigmaX=0)
    return img


# function to pixelate a face in an image
def pixelate_face(image, blocks=10):
    # get the height and width of the image
    h, w = image.shape[:2]
    # calculate the size of each block
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # calculate the starting and ending coordinates of the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the region of interest (ROI) from the image
            roi = image[startY:endY, startX:endX]
            # calculate the mean RGB value of the ROI
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            # draw a rectangle over the ROI with the mean RGB value
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    # return the pixelated image
    return image


# set the pixelation factor and start the video capture
factor = 8
cap = cv2.VideoCapture(0)

# load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# initialize the blur and pixelate toggles
blur_toggle = True
pixelate_toggle = True

# start the video capture loop
while 1:
    # read a frame from the video capture
    ret, frame = cap.read()
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    # loop over the detected faces
    for (x, y, w, h) in faces:
        # apply blur or pixelation to the face region based on the toggles
        if blur_toggle:
            frame[y:y + h, x:x + w] = blur(frame[y:y + h, x:x + w], factor)
        if pixelate_toggle:
            frame[y:y + h, x:x + w] = pixelate_face(frame[y:y + h, x:x + w], factor)
    # show the resulting frame
    cv2.imshow('Live', frame)
    # toggle the blur or pixelate effect if 'm' key is pressed
    if cv2.waitKey(1) == 27:
        break

# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()









































