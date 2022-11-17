import cv2
import numpy as np
def scnr(image):
    b, g, r = cv2.split(image)
    y, x, channels = image.shape

    for u in range(y):
        for v in range(x):
            max_rb = (r[u, v], b[u, v])
            print(max_rb)
