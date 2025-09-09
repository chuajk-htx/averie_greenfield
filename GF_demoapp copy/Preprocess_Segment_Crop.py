### Image Segmentation & Cropping ###

import cv2
import numpy as np
import math
import streamlit as st


def openfile(imagefilepath):
    # Load (IR) image of eye in grayscale
    image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE) 
    return image

""" see if arg is imagefile path or the image itself and amend accordingly
"""
# Check if item is an np.ndarray
#if isinstance(item, np.ndarray):
#    print("var_list is a NumPy array.")
#else:
#    print("var_list is not a NumPy array.")

def preprocess(image):
    # Preprocessing - grayscale, blurring/denoising, enhance contrast    
    # Smoothing/Denoising with Gaussian blurring 
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Contrast enhancement with CLAHE - <1.5 for unevenly lit image, ideal range 1.0-4.0, uniformly lit 2.0-3.0
    # Dynamically calculate clip limit
    mean_intensity = np.mean(image_blur)
    clip_limit = max(1.5, min(5.0, mean_intensity / 50))  # /50 because 0-255 /50 gives range of value between 0-5.1
    
    # clip_limit cap at max 3.0
    if clip_limit >= 3:
        clip_limit = 3.0
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(image_blur)

    return cl


def find_pupil(threshInv2, cl):

    # Find contour
    contours, _ = cv2.findContours(threshInv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #or RETR_TREE

    # Blank canvas to draw contours for pupil detection
    drawing = np.zeros((cl.shape[0], cl.shape[1], 3), dtype=np.uint8)

    # Draw on blank canvas
    cv2.drawContours(drawing, contours, -1, (0, 150, 0), 3) 

    largest = max(contours, key = cv2.contourArea) # Get largest contour

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, False), True)

        if cnt.shape == largest.shape:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            area = cv2.contourArea(cnt) # actual area
            circleArea = radius * radius * np.pi
            cv2.circle(drawing, (int(cx), int(cy)), math.ceil(radius), (255, 255, 0), cv2.FILLED)
            #print('center={},{}'.format(cx, cy))
            
            return area, cx, cy, radius, circleArea


import iris

class Segmenter():
    def __init__(self, image):
        self.image = image
        self.processed = None
        self.segmented = None
        self.cropped = None
        self.pupil_center_x = None
        self.pupil_center_y = None
        self.iris_center_x = None
        self.iris_center_y = None

    def process(self):
        self.processed = preprocess(self.image)
        return self

    def segment(self):
        
        # Create IRISPipeline object
        iris_pipeline = iris.IRISPipeline()

        # Perform inference
        output = iris_pipeline(img_data=self.processed, eye_side="right")

        segmap = iris_pipeline.call_trace['segmentation']

        eyeball_idx = segmap.index_of("eyeball")
        preds = segmap.predictions
        eyeball_mask = preds[..., eyeball_idx] >= 0.5  # Use appropriate threshold if needed; binary mask: white = keep, black = leaveout; outputs = True or False at each pixel 

        # Apply mask to the image: keep only eyeball region
        masked_img = np.zeros_like(self.processed)
        masked_img[eyeball_mask] = self.processed[eyeball_mask] # apply boolean mask array, selecting only the pixels in masked_img where eyeball_mask is True

        self.segmented = masked_img 

        return self


    def cropping(self):

        if self.segmented is not None:
            img = self.segmented
        else:
            img = self.processed

        # Create IRISPipeline object
        iris_pipeline = iris.IRISPipeline()

        # Perform inference
        output = iris_pipeline(img_data=img, eye_side="right")

        eye_center = iris_pipeline.call_trace['eye_center_estimation']

        if eye_center is None:
        #pupil_cx, pupil_cy = 
        #iris_cx, iris_cy = 

        #if eye_center.pupil_x is None and eye_center.pupil_y is None:
        #    if eye_center.iris_x is None and eye_center.iris_y is None:

            ret, thresh2 = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY) 
            threshInv2 = cv2.bitwise_not(thresh2)
            
            try:
                result = find_pupil(threshInv2, img)
                area, cx, cy, radius, circleArea = result

            except Exception as e:
                print('Error finding pupil/iris center')

        elif eye_center.pupil_x is None and eye_center.pupil_y is None:
            cx, cy = eye_center.iris_x, eye_center.iris_y

        else:
            cx, cy = eye_center.pupil_x, eye_center.pupil_y


        size = 175 
        #cl.shape[0] == y and cl.shape[1] == y:
        # normally 640x480
        original_size_x, original_size_y = img.shape[1], img.shape[0]

        exceed_y = original_size_y - 175
        adjusted_y = original_size_y - 350
        exceed_x = original_size_x - 175
        adjusted_x = original_size_x - 350


        if cx < 175 and cy < 175: # x, y too small
            cropped = img[0:350, 0:350].copy()
        elif cx <= exceed_x and cy < 175: # x ok, y too small:
            cropped = img[0:350, int(cx)-size:int(cx)+size ].copy()
        elif cx < 175 and cy <= exceed_y: # y ok, x too small:
            cropped = img[int(cy)-size:int(cy)+size, 0:350].copy()

        elif cx <= exceed_x and cy > exceed_y: # x ok, y too big:
            cropped = img[adjusted_y:original_size_y, int(cx)-size:int(cx)+size ].copy()
        elif cx > exceed_x and cy <= exceed_y: # y ok, x too big:
            cropped = img[int(cy)-size:int(cy)+size, adjusted_x:original_size_x].copy()

        elif cx < 175 and cy > exceed_y: # x too small, y too big
            cropped = img[adjusted_y:original_size_y, 0:350].copy()
        elif cx > exceed_x and cy < 175: # x too big, too small
            cropped = img[0:350, adjusted_x:original_size_x].copy() 

        else:
            cropped = img[ int(cy)-size:int(cy)+size, int(cx)-size:int(cx)+size ].copy()
    
        self.cropped = cropped

        return self
    