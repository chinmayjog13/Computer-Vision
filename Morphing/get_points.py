# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:38:30 2018

@author: CJog
"""

import cv2
import argparse

points = []

def get_points(event, x, y, flags, param):
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x,y), 2, (0,0,255), -1)
        points.append([x,y])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, 
                    help='Image filename without format, format should be .jpg')
    args = vars(ap.parse_args())
    filename = args['image']
    
    image = cv2.imread(filename + '.jpg', cv2.IMREAD_COLOR)
    h,w = image.shape[:2]
    file = open(filename + ".txt", 'w')
    points.append([0,0])
    points.append([int(w/2),0])
    points.append([int(w-1),0])
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_points, image)
    
    while True:
        cv2.imshow("image", image)
        
        # Select points by clicking left mouse button here, press c when all points have been selected
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            break
    
    points.append([0,int(h-1)])
    points.append([int(w-1),int(h-1)])
    
    for p in points:
        file.write(' '.join(str(t) for t in p) + '\n')
        cv2.circle(image, (p[0],p[1]), 2, (0,0,255), -1)
    file.close()
    #cv2.imwrite('image.jpg', image)
    cv2.imshow('captured points', image)
    cv2.waitKey(0)