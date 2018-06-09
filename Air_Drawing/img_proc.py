# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:56:50 2018

@author: CJog
"""

import cv2
import numpy as np

class image_processing:
    def __init__(self, frame, drawing, model):
        self.frame = frame
        self.drawing = drawing
        self.model = model
    
    def process_drawing(self):
        height, width = self.frame.shape[:2]
        
        #Create blank images
        image = np.zeros((height,width,3), np.uint8)
        prediction = np.zeros((100,320,3), np.uint8)
        
        #Make tracked motion locations white, to get a white foreground against a black background
        for [x, y] in self.drawing:
            cv2.rectangle(image, (x, y), 
                          (x-50, y+50), (255, 255, 255), -1)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 1)
        
        #Resize the image to the size of our dataset, so the model can classify accurately
        pred_image = cv2.resize(image, (20,20), cv2.INTER_AREA)
        
        #Initialise HOG descriptor
        hog = self.get_hog()
        descriptor = []
        
        #Compute HOG descriptor for test image
        descriptor.append(hog.compute(pred_image))
        descriptor = np.squeeze(descriptor)
        ans = self.predict(descriptor, prediction)
        return ans
    
    def predict(self, feature_vector, ans):
        #Predict digit using trained model
        pred = self.model.predict(feature_vector.reshape(1,-1))
        cv2.putText(ans, "digit is= {}".format(str(pred)), (25,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3)
        return ans
    
    def deskew(img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img
    
    def get_hog(self) : 
        winSize = (20,20)
        blockSize = (16,16)
        blockStride = (4,4)
        cellSize = (16,16)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True
        
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
        
        return hog

class MotionDetector:
    def __init__(self, acc_weight=0.5, delta_threshold=5, area=3000, min_area=2000, max_area=2500):
        self.acc_weight = acc_weight
        self.delta_threshold = delta_threshold
        self.area = area
        self.min_area = min_area
        self.max_area = max_area
        self.avg = None
    
    def large_update(self, frame):
        loc = []
        
        if self.avg is None:
            self.avg = frame.astype("float")
            return loc
        
        cv2.accumulateWeighted(frame, self.avg, self.acc_weight)
        frame_delta = cv2.absdiff(frame, cv2.convertScaleAbs(self.avg))
        
        threshold = cv2.threshold(frame_delta, self.delta_threshold, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)
        
        contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1]
        
        for i in contours:
            if cv2.contourArea(i) > self.area:
                loc.append(i)
        return loc
    
    def small_update(self, frame):
        loc = []
        
        if self.avg is None:
            self.avg = frame.astype("float")
            return loc
        
        cv2.accumulateWeighted(frame, self.avg, self.acc_weight)
        frame_delta = cv2.absdiff(frame, cv2.convertScaleAbs(self.avg))
        
        threshold = cv2.threshold(frame_delta, self.delta_threshold, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)
        
        contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1]
        
        for i in contours:
            if cv2.contourArea(i) > self.min_area and cv2.contourArea(i) < self.max_area:
                loc.append(i)
        return loc