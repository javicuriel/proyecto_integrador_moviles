import numpy as np
import cv2
from matplotlib import pyplot as plt


def normal(img_path):
    # 184 pixeles = 60um
    # 60/184 = 0.33 um/perPixel
    original = cv2.imread(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Make background the foreground
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    # Dilate the foreground (==background)
    thresh = cv2.dilate(thresh,kernel,iterations=7)
    # Invert blacks and whites to set foreground as background
    thresh = cv2.bitwise_not(thresh)
    # Find Countours
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    # Get last Y in contours (closest to origin 0,0)
    _,y,_,_ = cv2.boundingRect(contours[-1])
    # Set maximum allowed Y (10%)
    maxy = y*1.1
    # Total area
    total = 0
    # Find countours with range (0 -> maxy)
    for c in reversed(contours):
        _,y,_,_ = cv2.boundingRect(c)
        if(y > maxy): break
        total += cv2.contourArea(c)
        cv2.drawContours(original, c, -1, (255,0,0), 5)

    print("Area: " + str(total*0.33) + " um^2")


def median():
    img = cv2.imread(img_path,0)
    median = cv2.medianBlur(img,9)
    ret,thresh = cv2.threshold(median,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.resize(thresh, (900,900))
    countour(thresh)

def bilat():
    img = cv2.imread(img_path,0)
    blur = cv2.bilateralFilter(img,9,75,75)
    ret ,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.resize(thresh, (900,900))
    countour(thresh)

def countour(image):
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    # cv2.drawContours(original, contours, -1, (255,0,0), 3)

    # c = max(contours, key = cv2.contourArea)
    # sorted_c = sorted(contours, key = cv2.contourArea)

    color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # Get last Y in contours (closest to origin 0,0)
    _,y,_,_ = cv2.boundingRect(contours[-1])
    # Set maximum allowed Y (10%)
    maxy = y*1.1
    # Find countours with range (0 -> maxy)
    for c in reversed(contours):
        _,y,_,_ = cv2.boundingRect(c)
        if(y > maxy): break
        cv2.drawContours(color, c, -1, (255,0,0), 3)


    print(len(contours))
    cv2.imshow("window", color)
    # cv2.imshow("window", original)
    cv2.waitKey()

# normal()
# bilat()
# median()


normal('vanadio2.jpg')
normal('vanadio4.jpg')
normal('vanadio6.jpg')
normal('vanadio8.jpg')
