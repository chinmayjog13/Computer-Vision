# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:15:33 2018

@author: CJog
"""

import cv2
import numpy as np
from imutils import face_utils
import dlib

def inRect(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def draw_delaunay(image, subdiv, color):
    tlist = subdiv.getTriangleList()
    size = image.shape
    r = (0,0, size[1], size[0])
    for t in tlist:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])
        
        if inRect(r, p1) and inRect(r, p2) and inRect(r, p3):
            cv2.line(image, p1, p2, color, 1)
            cv2.line(image, p2, p3, color, 1)
            cv2.line(image, p3, p1, color, 1)

def write_triangles(size, subdiv, points):
    file = open('trilist.txt', 'w')
    r = (0,0, size[1], size[0])
    tlist = subdiv.getTriangleList()
    for t in tlist:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])
        if inRect(r, p1) and inRect(r, p2) and inRect(r, p3):
            tri = []
            tri.append([points.index(p1), points.index(p2), points.index(p3)])
            for p in tri:
                file.write(' '.join(str(t) for t in p) + '\n')
    file.close()
        
if __name__ == '__main__':  
    img1 = cv2.imread('batman_keaton.jpg')
    img2 = cv2.imread('batman_affleck.jpg')
    print('image reading done')
    
    alpha = 0.5
    h,w = img1.shape[:2]
    subdiv = cv2.Subdiv2D((0,0,w,h))
    
    points1 = []
    points2 = []
    points = []
    file1 = open('batman_keaton.txt', 'r')
    file2 = open('batman_affleck.txt', 'r')
    
    for line in file1:
        x, y = line.split()
        points1.append((int(x), int(y)))
    
    for line in file2:
        x, y = line.split()
        points2.append((int(x), int(y)))
    
    for i in range(0, len(points1)):
        x = (1-alpha)*points1[i][0] + alpha*points2[i][0]
        y = (1-alpha)*points1[i][1] + alpha*points2[i][1]
        points.append((x,y))
    
    for p in points:
        subdiv.insert(p)
    
    # Optional lines to visualize delaunay triangles
    draw_delaunay(img1, subdiv, (0,0, 255))
    draw_delaunay(img2, subdiv, (0,0, 255))
    print('Delaunay Triangles drawn')
    
    # Store triangle vertices in text file
    size = [h, w]
    write_triangles(size, subdiv, points)
    print('triangles written to text')
    
    cv2.imshow('keaton', img1)
    cv2.imshow('affleck', img2)
    cv2.waitKey(0)