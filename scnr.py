import cv2
import numpy as np
class SCNR:
    def __init__(self, image, amount=0.5):
        self.image = image
        self.amount = amount
        self.x, self.y, channels = self.image.shape

    def average_neutral(self):
        b, g, r = cv2.split(self.image)
        for u in range(self.x):
            for v in range(self.y):
                m = 0.5 * (r[u, v] + b[u, v])
                g[u, v] = min(g[u , v], m)

        return cv2.merge((b, g, r))

    def maximum_neutral(self):
        b, g, r = cv2.split(self.image)
        for u in range(self.x):
            for v in range(self.y):
                m = max(r[u, v], b[u, v])
                g[u, v] = min(g[u , v], m)

        return cv2.merge((b, g, r))

    def additive_mask(self):
        b, g, r = cv2.split(self.image)
        for u in range(self.x):
            for v in range(self.y):
                m = min(1, r[u, v] + b[u, v])
                g[u, v] = g[u, v] * (1 - self.amount) + (1-m) + m*g[u, v]

        return cv2.merge((b, g, r))

    def maximum_mask(self):
        b, g, r = cv2.split(self.image)
        for u in range(self.x):
            for v in range(self.y):
                m = max(r[u, v] , b[u, v])
                g[u, v] = g[u, v] * (1 - self.amount) * (1-m) + m*g[u, v]

        return cv2.merge((b, g, r))
