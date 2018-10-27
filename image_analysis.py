import numpy as np
import PIL.Image
import cv2


def findAverageCountourSize(contours):
    area = 0
    for c in contours:
        area += cv2.contourArea(c)
    if area:
        area = area/len(contours)

    return area

def countWhites(image):
    count = 0
    for x in image:
        for i in x:
            if i == 255:
                count += 1
    return count

def analyseImage(images_src):
    # Get image and convert it to array
    image = np.array(PIL.Image.open(images_src))
    imageColor = np.copy(image)
    # Up threshold for beter detection of features
    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
    # Convert to Grayscale for countour detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find Countours
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    # Get average size of countours
    avg_area = findAverageCountourSize(contours)
    white_pixel_count = countWhites(image)
    # Draw contours in image
    cv2.drawContours(imageColor, contours, -1, (255,0,0), 3)
    # Write to file to see countours
    # cv2.imwrite('cv2_'+images_src,imageColor)
    # To see picture with contours
    # cv2.imshow("window title", imageColor)
    # cv2.waitKey()
    return len(contours), avg_area, white_pixel_count


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


print(analyseImage('vanadio2.jpg'))
# print(analyseImage('vanadio4.jpg'))
# print(analyseImage('vanadio6.jpg'))
# print(analyseImage('vanadio8.jpg'))
