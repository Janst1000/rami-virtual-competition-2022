import cv2
import os
from pprint import pprint
import numpy as np

dataset = "./rami_marine_dataset/class_1/red"
img_set = os.listdir(dataset)
img_set.sort()

lower = np.array([0, 0, 0])
upper = np.array([65, 255, 255])

for file in img_set:
    f = os.path.join(dataset, file)
    img = cv2.imread(f)
    img = cv2.resize(img, (1280, 720))
    print(file)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    threshhold_img = cv2.inRange(hsv_img, lower, upper)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1= 45, param2= 50, maxRadius=200)

    if circles is not None:
        print("Found circle")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)

    #cv2.imshow("image", threshhold_img)
    cv2.imshow("original", img)
    key = cv2.waitKey(1)
    # 'q' to stop
    if key == ord('q'):
        break
    # Print key 
    elif key != -1:
        print(key)
