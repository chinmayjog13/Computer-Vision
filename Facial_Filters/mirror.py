# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:59:02 2018

@author: CJog
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

while True:
    ret, frame = cap.read()
    frame2 = frame[:,:int(w/2)]
    frame2 = cv2.flip(frame2, 1)
    frame[:,int(w/2):] = frame2
    
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()