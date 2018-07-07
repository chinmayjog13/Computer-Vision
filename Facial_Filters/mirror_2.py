# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:07:31 2018

@author: CJog
"""

import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

p1 = np.array(([0,0],[int(h/2)-1,0],[int(h/2)-1, int(h/2)-1]), dtype=np.int32)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (h,h))
    s1 = np.zeros((int(h/2), int(h/2),3), dtype='uint8')
    s1 = cv2.fillPoly(s1, [p1], (255,255,255))
    
    s1 = s1 & frame[:int(h/2),:int(h/2)]
    s2 = imutils.rotate(s1, 270)
    s3 = imutils.rotate(s1, 180)
    s4 = imutils.rotate(s1, 90)
    
    m2 = cv2.flip(s1, 1)
    m3 = cv2.flip(s2, 0)
    m4 = cv2.flip(s3, 1)
    m1 = cv2.flip(s4, 0)
    
    m2 = cv2.add(s2,m2)
    m3 = imutils.rotate(m2, 270)
    m4 = imutils.rotate(m2, 180)
    m1 = imutils.rotate(m2, 90)
    
    frame[:int(h/2),:int(h/2)] = m1
    frame[:int(h/2),int(h/2):] = m2
    frame[int(h/2):,int(h/2):] = m3
    frame[int(h/2):,:int(h/2)] = m4
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()