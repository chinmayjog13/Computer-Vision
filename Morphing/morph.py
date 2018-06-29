# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:24:01 2018

@author: CJog
"""

import cv2
import numpy as np

def read_points(filename):
    points = []
    file = open(filename, 'r')
    for line in file:
        x, y = line.split(sep=' ')
        points.append((int(x), int(y)))
    return points

def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph(img1, img2, output, t1, t2, t, alpha):
    # Create bounding rectangles around triangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    t1Rect = []
    t2Rect = []
    tRect = []
    
    for i in range(0,3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
    
    # Create mask to keep only triangular regions
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    size = (r[2], r[3])
    
    # Warp image patches
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    
    # Alpha blend image patches
    imgRect = (1.0-alpha)*warpImage1 + alpha*warpImage2
    
    # Paste warped triangular regions in final output
    output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]*(1-mask) + imgRect*mask

if __name__ == '__main__':
    # List of images to be morphed in sequential order
    file = ['keaton', 'batman_keaton', 'batman_bale', 'bale', 'affleck', 'batman_affleck']
    
    # Loop over all images morphing two at a time
    for j in range(len(file)-1):
        # Read points from text file
        points1 = read_points(file[j] + '.txt')
        points2 = read_points(file[j+1] + '.txt')
        img1 = cv2.imread(file[j] + '.jpg')
        img2 = cv2.imread(file[j+1] + '.jpg')
        
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        
        # Loop over different values of alpha for animation effect
        for alpha in np.linspace(0,10,25):
            alpha = alpha*0.1
            points=[]
            for i in range(0, len(points1)):
                x = (1-alpha)*points1[i][0] + alpha*points2[i][0]
                y = (1-alpha)*points1[i][1] + alpha*points2[i][1]
                points.append((x,y))
            
            output = np.zeros(img1.shape, dtype=img1.dtype)
            
            # Read triangle vertices corresponding to Delaunay triangles from text file
            tri = open('trilist.txt', 'r')
            for line in tri:
                x, y, z = line.split()
                x = int(x)
                y = int(y)
                z = int(z)
                
                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [points[x], points[y], points[z]]
                
                # Morph each triangle present in list
                morph(img1, img2, output, t1, t2, t, alpha)
            
            # Different delays for user to see each end result before morphing to another image
            # and give a smooth animation
            cv2.imshow('Morphed', np.uint8(output))
            if alpha == 0 and j == 0:
                cv2.waitKey(1000)
            if j == 4 and alpha == 1:
                cv2.waitKey(0)
            if alpha == 1:
                cv2.waitKey(500)
            else:
                cv2.waitKey(50)
