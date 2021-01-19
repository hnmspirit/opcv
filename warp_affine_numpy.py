import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('home.jpg')
h,w = im.shape[:2]

# transform matrix: rotate -10 deg
phi = -10*pi/180
M = np.float32([[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
M1 = np.linalg.inv(M) # inverse transform

# grid
y,x = np.indices((h, w))
pts = np.vstack((x.ravel(),y.ravel(),np.ones(y.size)))
pts = M1.dot(pts).reshape(3,h,w) # apply transform
x,y = pts[:2]

# border idx
nonx = (x < 0) | (x > w-1)
nony = (y < 0) | (y > h-1)

# interpolate
x0 = np.floor(x).astype(np.int64)
x1 = np.ceil(x).astype(np.int64)
y0 = np.floor(y).astype(np.int64)
y1 = np.ceil(y).astype(np.int64)

x0 = np.clip(x0, 0, w-1) # clip idx out of bound
x1 = np.clip(x1, 0, w-1)
y0 = np.clip(y0, 0, h-1)
y1 = np.clip(y1, 0, h-1)

p1 = im[y0,x0]
p2 = im[y1,x0]
p3 = im[y0,x1]
p4 = im[y1,x1]

maskx = np.float64(x1 == x) # no-interpolate idx
masky = np.float64(y1 == y)

k1 = np.maximum(((x1-x) * (y1-y)), maskx, masky)[...,None] # coef
k2 = ((x1-x) * (y-y0))[...,None]
k3 = ((x-x0) * (y1-y))[...,None]
k4 = ((x-x0) * (y-y0))[...,None]

dt = np.uint8(k1*p1 + k2*p2 + k3*p3 + k4*p4)

# border draw
dt[nony] = 0
dt[nonx] = 0

# warpAffine by lib
dt0 = cv2.warpAffine(im, M[:2], (w,h))

# vis
g = plt.figure(figsize=(9,3))
plt.subplot(131)
plt.imshow(im[...,::-1])
plt.subplot(132)
plt.imshow(dt[...,::-1])
plt.subplot(133)
plt.imshow(dt0[...,::-1])
plt.show()