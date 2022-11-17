import cv2
import os
from pprint import pprint
from cv2 import threshold
import numpy as np
import skimage.exposure
from scnr import scnr

dataset = "./rami_marine_dataset/class_2/number_2"
img_set = os.listdir(dataset)
img_set.sort()

def remove_tint(img):
    # convert to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # separate channels
    h,s,v = cv2.split(hsv)

    # reverse the hue channel by 180 deg out of 360, so in python add 90 and modulo 180
    h_new = (h + 120) % 180

    # combine new hue with old sat and value
    hsv_new = cv2.merge([h_new,s,v])

    # convert back to BGR
    bgr_new = cv2.cvtColor(hsv_new,cv2.COLOR_HSV2BGR)

    # Get the average color of bgr_new
    ave_color = cv2.mean(bgr_new)[0:3]
    print(ave_color)

    # create a new image with the average color
    color_img = np.full_like(img, ave_color)

    # make a 50-50 blend of img and color_img
    blend = cv2.addWeighted(img, 0.5, color_img, 0.5, 0.0)

    # stretch dynamic range
    result = skimage.exposure.rescale_intensity(blend, in_range='image', out_range=(0,255)).astype(np.uint8)
    return result

for file in img_set:
    f = os.path.join(dataset, file)
    img = cv2.imread(f)
    img = cv2.resize(img, (720, 405))
    print(file)
    
    scnr(img)

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # remove teal tint from image
    ret = remove_tint(img)

    # threshold grayscale image
    gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # invert thresholdled image
    thresh = cv2.bitwise_not(thresh)
    
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hconcat_img = np.hstack((img, gray, thresh))
    cv2.imshow("original", hconcat_img)
    key = cv2.waitKey(1)
    # 'q' to stop
    if key == ord('q'):
        break
    # Print key 
    elif key != -1:
        print(key)
