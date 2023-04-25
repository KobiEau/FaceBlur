import cv2
import numpy as np
def blur(img,k):
    h,w = img.shape[:2]
    kh,kw = h//k,w//k
    if kh%2==0:
        kh-=1
    if kw%2==0:
        kw-=1
    img = cv2.GaussianBlur(img,ksize=(kh,kw),sigmaX=0)
    return img
def pixelate_face(image, mask, blocks=10):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # apply the pixelation effect only to the face pixels
            if np.all(mask[startY:endY, startX:endX]):
                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (B, G, R), -1)
    # return the pixelated blurred image
    return image

factor = 6
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while 1:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    # create an empty mask with the same size as the input frame
    mask = np.zeros_like(gray)
    # loop over the detected faces and set the corresponding pixels in the mask to 1
    for (x,y,w,h) in faces:
        mask[y:y+h, x:x+w] = 1
    # apply the pixelation effect to the face pixels only
    pixelated_frame = pixelate_face(frame, mask, blocks=10)
    # display the pixelated frame
    cv2.imshow('Live', pixelated_frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

