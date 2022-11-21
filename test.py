import cv2
import os
from pprint import pprint
from cv2 import threshold
import numpy as np
import skimage.exposure
from scnr import SCNR
from matplotlib import pyplot as plt

dataset = "./rami_marine_dataset/class_2/number_3"
img_set = os.listdir(dataset)
img_set.sort()

def remove_tint(img):
    # convert to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # separate channels
    h,s,v = cv2.split(hsv)

    # reverse the hue channel by 180 deg out of 360, so in python add 90 and modulo 180
    h_new = (h + 90) % 180

    # combine new hue with old sat and value
    hsv_new = cv2.merge([h_new,s,v])

    # convert back to BGR
    bgr_new = cv2.cvtColor(hsv_new,cv2.COLOR_HSV2BGR)

    # Get the average color of bgr_new
    ave_color = cv2.mean(bgr_new)[0:3]

    # create a new image with the average color
    color_img = np.full_like(img, ave_color)

    # make a 50-50 blend of img and color_img
    blend = cv2.addWeighted(img, 0.5, color_img, 0.5, 0.0)

    # stretch dynamic range
    result = skimage.exposure.rescale_intensity(blend, in_range='image', out_range=(0,255)).astype(np.uint8)
    return result

def concat_images(*images):
    # convert image to bgr if necassary
    images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in images]
    hconcat_img = np.hstack(images)
    return hconcat_img

def process_image(image, contrast=False):
 
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # increase contrast
    if contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        blur = clahe.apply(blur)

    thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # invert thresholdled image
    thresh = cv2.bitwise_not(thresh)
    
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    #gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return thresh

def remove_green(img, percent = 0.5):
        # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # desaturate
    s_desat = cv2.multiply(s, percent).astype(np.uint8)
    hsv_new = cv2.merge([h,s_desat,v])
    bgr_desat = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    # create 1D LUT for green
    # (120 out of 360) = (60 out of 180)  +- 25
    lut = np.zeros((1,256), dtype=np.uint8)
    white = np.full((1,50), 255, dtype=np.uint8)
    lut[0:1, 35:85] = white
    print(lut.shape, lut.dtype)

    # apply lut to hue channel as mask
    mask = cv2.LUT(h, lut)
    mask = mask.astype(np.float32) / 255
    mask = cv2.merge([mask,mask,mask])

    # mask bgr_desat and img
    result = mask * bgr_desat + (1 - mask)*img
    result = result.clip(0,255).astype(np.uint8)
    return result

def auto_canny(image, sigma = 0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) *v))
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out = cv2.VideoWriter('number3.avi',fourcc, 20, (1440, 405))
    for file in img_set:
        f = os.path.join(dataset, file)
        img = cv2.imread(f)
        img = cv2.resize(img, (720, 405))
        print(file)
        
        
        green_removed = remove_green(img, 25)
        
        ret = remove_tint(green_removed)

        h, s, v = cv2.split(cv2.cvtColor(ret, cv2.COLOR_BGR2HSV))
        # increase contrast in s
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        s = clahe.apply(s)
        
        # merge h back to hsv
        hsv = cv2.merge([h,s,v])

        img_copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # get lab decomp
        lab = cv2.cvtColor(img_copy, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

            
        # max between l and b
        max_l_b = cv2.max(l, b)

        
        # doing normalization
        normed = cv2.normalize(max_l_b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hconcat_img = concat_images(img, normed)
        cv2.imshow("original", hconcat_img)
        
        # save all frames as a video
        #hconcat_img = cv2.resize(hconcat_img, (720, 405))
        out.write(hconcat_img)
        print(hconcat_img.shape)

        key = cv2.waitKey(1)
        # 'q' to stop
        if key == ord('q'):
            out.release()
            break
        # Print key 
        elif key != -1:
            print(key)
    #out.release()
