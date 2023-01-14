import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import numpy as np

import sys
sys.path.insert(0, './models')

class Detector:
    def __init__(self, weights, imgsz=640, trace=False, half=False, augment=False, conf_thres = 0.25, iou_thres = 0.45, device='cpu'):
        self.weights = weights
        self.trace = trace
        self.half = half
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if torch.cuda.is_available() and device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())
        self.shape = (imgsz, imgsz)
        self.imgsz = check_img_size(640, s=self.stride)
        if trace:
            self.model = TracedModel(self.model, self.device, 640)

        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Colors
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
                

    def detect(self, img, trace=False, augment=False, conf_thres = 0.25, iou_thres = 0.45, show_all = True):
        # Resize image for yolo model
        img = cv2.resize(img, (640, 640))

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time_synchronized()
        im0 = img.copy()
        # Padded resize
        img = letterbox(img, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        im0s = im0
        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            s, im0, frame =  '', im0s, getattr(img, 'frame', 0)


            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
                if show_all == True:
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # scaling xywh to original image size
                        xywh[0] = int(xywh[0] * self.shape[1])
                        xywh[1] = int(xywh[1] * self.shape[0])
                        xywh[2] = int(xywh[2] * self.shape[1])
                        xywh[3] = int(xywh[3] * self.shape[0])

                        #xyxy_scaled = xyxy * torch.Tensor([self.shape[1], self.shape[0], self.shape[1], self.shape[0]]).to(self.device)
                        

                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        line = (self.names[int(cls)], *xywh, f'{conf:.2f}')
                        
                        print(f'{conf:.2f}')
                        if conf >= 0.75:
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                            return im0, line
                        else:
                            return im0, None
                else:
                    # get best match
                    best_match = det[0]
                    # show best match
                    *xyxy, conf, cls  = best_match
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # scaling xywh to original image size
                    xywh[0] = int(xywh[0] * self.shape[1])
                    xywh[1] = int(xywh[1] * self.shape[0])
                    xywh[2] = int(xywh[2] * self.shape[1])
                    xywh[3] = int(xywh[3] * self.shape[0])

                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    line = (self.names[int(cls)], *xywh, f'{conf:.2f}')
                    
                    print(f'{conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    return im0, line
        return im0, None
