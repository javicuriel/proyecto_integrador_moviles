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



print(analyseImage('vanadio2.jpg'))
# print(analyseImage('vanadio4.jpg'))
# print(analyseImage('vanadio6.jpg'))
# print(analyseImage('vanadio8.jpg'))
