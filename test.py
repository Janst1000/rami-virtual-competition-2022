import cv2
import os
from pprint import pprint
from cv2 import threshold
import numpy as np

dataset = "./rami_marine_dataset/class_3/number_6"
img_set = os.listdir(dataset)
img_set.sort()

#lower = np.array([0, 0, 0])
#upper = np.array([65, 255, 255])

for file in img_set:
    f = os.path.join(dataset, file)
    img = cv2.imread(f)
    img = cv2.resize(img, (720, 405))
    print(file)
    #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #threshhold_img = cv2.inRange(hsv_img, lower, upper)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    red_image = img[:,:,2]
    threshold_image = cv2.threshold(red_image, 75, 255, cv2.THRESH_TOZERO)[1]
    canny_edge = cv2.Canny(threshold_image, 150, 200)
    hconcat_img = np.hstack((threshold_image, canny_edge))




    #cv2.imshow("image", threshhold_img)
    cv2.imshow("original", hconcat_img)
    key = cv2.waitKey(1)
    # 'q' to stop
    if key == ord('q'):
        break
    # Print key 
    elif key != -1:
        print(key)
