# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:16:33 2018

@author: CJog
"""

import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

h, w = frame.shape[:2]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        center = shape[28][0], shape[28][1]
        radius = int(dist.euclidean(shape[28], shape[8]))
        
        f_mask = np.zeros((h,w,3), np.uint8)
        cv2.circle(f_mask, center, radius, (255,255,255), -1)
        b_mask = cv2.bitwise_not(f_mask)
        
        b_mask = frame & b_mask
        f_mask = frame & f_mask
        
        frame = cv2.GaussianBlur(b_mask, (21,21), 0)
        frame = cv2.add(frame, f_mask)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release