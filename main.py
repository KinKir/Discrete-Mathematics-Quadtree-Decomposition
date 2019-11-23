import numpy as np
import cv2

def showImage(imageName):
    cv2.imshow("Lenna",imageName)
    cv2.waitKey(0)

def quadtreeDecomposition(image, height, width, n):
    if(n==0):
        m,_ = cv2.meanStdDev(image)
        m = tuple(m.flatten())
        return m
    else:
        quadtreeImage = np.zeros((height,width,3), np.uint8)

        halfHeight = height//2
        halfWidth = width//2

        # Northwest
        imageNW = image[0:halfHeight,0:halfWidth]
        quadtreeImage[0:halfHeight,0:halfWidth] = quadtreeDecomposition(imageNW, halfHeight, halfWidth, n-1)

        # Northeast
        imageNE = image[0:halfHeight,halfWidth:width]
        quadtreeImage[0:halfHeight,halfWidth:width] = quadtreeDecomposition(imageNE, halfHeight, halfWidth, n-1)

        # Southeast
        imageSE = image[halfHeight:height,halfWidth:width]
        quadtreeImage[halfHeight:height,halfWidth:width] = quadtreeDecomposition(imageSE, halfHeight, halfWidth, n-1)

        # Southwest
        imageSW = image[halfHeight:height,0:halfWidth]
        quadtreeImage[halfHeight:height,0:halfWidth] = quadtreeDecomposition(imageSW, halfHeight, halfWidth, n-1)

        return quadtreeImage


imageInput = cv2.imread("./images/airplane.png")

height, width = imageInput.shape[0], imageInput.shape[1]

for i in range(1,8):
    blank_image = quadtreeDecomposition(imageInput,height,width,i)
    showImage(blank_image)


