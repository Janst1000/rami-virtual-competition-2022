import cv2
import os
from pprint import pprint
import numpy as np
import skimage.exposure
from scnr import SCNR
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import Detection

#dataset = "./rami_marine_dataset/class_5/"
#img_set = os.listdir(dataset)
#img_set.sort()


"""
    This function was found here https://stackoverflow.com/questions/70876252/how-to-do-color-cast-removal-or-color-coverage-in-python
    it works pretty good but we modified it to stretch the dynamic range at the end
"""
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

""" 
    This function was found herehttps://stackoverflow.com/questions/64762020/how-to-desaturate-one-color-in-an-image-using-opencv2-in-python
"""
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
    #print(lut.shape, lut.dtype)

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
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input directory", required=True)
    parser.add_argument("-o", "--output", dest="output", help="create output video", required=False)
    dataset = parser.parse_args().input
    output = parser.parse_args().output
    img_set = os.listdir(dataset)
    img_set.sort()
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    if output != None:
        out = cv2.VideoWriter('example.avi',fourcc, 5, (1440, 405))
    numberDetector = Detection.Detector(weights="./models/numbersV2.pt", imgsz=640, half=False, device="cuda")
    ballDetector = Detection.Detector(weights="./models/ballsV1.pt", imgsz=640, half=False, device="cuda")

    # clear output file
    with open("output.txt", "w") as f:
        f.write("")
    
    # remove everything that isn't a jpg or png
    img_set = [x for x in img_set if x.endswith(".png")]
    # create key_list
    key_list = []
    for file in img_set:
        key_list.append(int(file.split(".")[0]))
    key_list.sort()
    img_set = [str(x) + ".png" for x in key_list]
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
        normed_bgr = cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR)

        detected, bounding_box = numberDetector.detect(img = normed_bgr, show_all=False, conf_thres=0.75)
        if bounding_box == None:
            detected, bounding_box = ballDetector.detect(img = img, show_all=False)
            if bounding_box == None:
                print("No ball detected")
                with open("output.txt", "a") as f:
                    f.write("\n")
            else:
                ball, x, y, w, h, cls = bounding_box
                centroid = (x, y)
                cv2.circle(detected, centroid, 5, (0, 0, 255), -1)
                print("Ball detected: ", ball, x, y, w, h, cls)
                with open("output.txt", "a") as f:
                    color = ball.split("_")[1]
                    f.write(color + "\n")

        else:
            #print(bounding_box)
            number, x, y, w, h, cls = bounding_box
            centroid = (x, y)
            #print("Number detected: ", number, "Centriod: ", centroid)
            cv2.circle(detected, centroid, 5, (0, 0, 255), -1)
            print("Number detected: ", number, x, y, w, h, cls)
            with open("output.txt", "a") as f:
                number = number.split("_")[1]
                f.write("number_" + number + "\n")
        detected = cv2.resize(detected, (720, 405))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hconcat_img = concat_images(img, detected)
        vconcat_img = concat_images(img, detected)
        cv2.imshow("original", hconcat_img)
        
        # save all frames as a video
        #hconcat_img = cv2.resize(hconcat_img, (720, 405))
        if output != None:
            out.write(vconcat_img)
        #    cv2.imwrite(output + file, detected)

        key = cv2.waitKey(1)
        # 'q' to stop
        if key == ord('q'):
            out.release()
            break
        # Print key
        elif key != -1:
            print(key)
    #out.release()
    # calculate accuracy
    with open("output.txt", "r") as f:
        preds = f.readlines()
    with open(dataset + "/ground_truth.txt", "r") as f:
        ground_truth = f.readlines()
    correct = 0
    total = len(ground_truth)
    for i in range(len(preds)):
        if preds[i] == ground_truth[i]:
            correct += 1
    accuracy = correct / total
    print("Accuracy: ", accuracy)