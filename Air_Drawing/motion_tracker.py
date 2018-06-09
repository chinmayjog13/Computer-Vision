# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:56:26 2018

@author: CJog
"""

import numpy as np
import cv2
import motion_Detection as md
import img_proc
import pickle

#select your laptop webcam for your video stream
cap = cv2.VideoCapture(0)

#Initialise motion detection class
motion = img_proc.MotionDetector()

#load the pre-trained svm model. You can train your model using train_digits.py
model = pickle.load(open('model_pickle.pkl', 'rb'))
descriptor = []
prev_frame = []
total = 0
clear = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    clear = clear + 1
    
    #Convert frame to grayscale and blur ro remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    #detect motion location
    loc = motion.large_update(gray)
    
    #use first 32 frames to create stable background estimation
    if total < 32:
        total = total + 1
        continue
    
    #if motion detected
    if len(loc) > 0:
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        
        for i in loc:
            (x, y, w, h) = cv2.boundingRect(i)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))
                
        #store motion locations of 80 previous frames
        if len(prev_frame) < 80:
            prev_frame.append([maxX, minY])
        else:
            del prev_frame[0]
            prev_frame.append([maxX, minY])
        clear = 0

    
    # Draw tracked motion for locations buffer
    if len(prev_frame) > 0:
        for j in range(len(prev_frame)):
                cv2.rectangle(frame, (prev_frame[j][0], prev_frame[j][1]), 
                          (prev_frame[j][0]-50, prev_frame[j][1]+50), (255, 0, 255), -1)
    
    #If no motion for 20 frames and sufficient motion has been tracked,
    # assume digit has been drawn and proceed to classify
    if clear > 20 and len(prev_frame) > 20:
        proc = img_proc.image_processing(frame, prev_frame, model)
        ans = proc.process_drawing()
        
        #show predicted digit
        cv2.imshow('prediction', ans)
        
        #Read video object again to clear frame
        ret, frame = cap.read()
        
        #clear locations buffer
        prev_frame = []
        
    #Show resulting frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    
    #Press 'q' key to terminate program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()