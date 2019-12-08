import quadtree as q
import cv2
import numpy as np

q.quadtreeSegmentation("lenna.png", limit=8, stdLimit=7.0)

# for i in np.arange(2.5, 17.5, 2.5):
#     q.quadtreeSegmentation("liechestein.png", 7, i)

# image1 = cv2.imread("./images/lenna.png")
# image2 = cv2.imread("./images/airplane.png")
# Q1 = q.quadtree(image1, 512, 512)
# Q2 = q.quadtree(image2, 512, 512)

# QR = q.imageUnion(Q1,Q2,8,0.7)

# cv2.imwrite("./result/union5050.png", QR)

