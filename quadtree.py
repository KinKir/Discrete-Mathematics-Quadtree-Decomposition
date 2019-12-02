import cv2
import numpy as np
import os

class quadtree: 
    def __init__(self, image, height, width):

        # Set mean and STD of image region
        self.mean, self.std = cv2.meanStdDev(image)
        self.mean = tuple(self.mean.flatten())

        # Set height and width of image region
        self.height = height
        self.width = width

        # Set half of height and width
        self.halfHeight = self.height//2
        self.halfWidth = self.width//2

        # Set image
        self.imageNW = None
        self.imageNE = None
        self.imageSE = None
        self.imageSW = None

        if(height!=1 and width!=1):
            self.imageNW = quadtree(image[0:self.halfHeight, 0:self.halfWidth], self.halfHeight, self.halfWidth)
            self.imageNE = quadtree(image[0:self.halfHeight, self.halfWidth:width], self.halfHeight, self.halfWidth)
            self.imageSE = quadtree(image[self.halfHeight:height, self.halfWidth:width], self.halfHeight, self.halfWidth)
            self.imageSW = quadtree(image[self.halfHeight:height, 0:self.halfWidth], self.halfHeight, self.halfWidth)

    def printTree(self, level):
        print(self.std)
        if(self.imageNW!=None or self.imageNE!=None or self.imageSE!=None or self.imageSW!=None):
            print(level, '-- NW ',end='')
            self.imageNW.printTree(level+1)
            print(level, '-- NE ',end='')
            self.imageNE.printTree(level+1)
            print(level, '-- SE ',end='')
            self.imageSE.printTree(level+1)
            print(level, '-- SW ',end='')
            self.imageSW.printTree(level+1)

    def createLeveledCompression(self, image, level):
        resultHeight = 1<<level
        resultWidth = 1<<level
        imageResult = np.zeros((resultHeight,resultWidth,3), np.uint8)

        if(level==0):
            return image.mean
        else:
            imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = image.createLeveledCompression(image.imageNW, level-1)
            imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = image.createLeveledCompression(image.imageNE, level-1)
            imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = image.createLeveledCompression(image.imageSE, level-1)
            imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = image.createLeveledCompression(image.imageSW, level-1)

        return imageResult 

    def scrambleImage(self, level):
        resultHeight = self.height
        resultWidth = self.width
        imageResult = np.zeros((resultHeight, resultWidth, 3), np.uint8)

        if(level==0):
            return self.mean
        elif(level%2==0):
            imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = self.imageSE.scrambleImage(level-1)
            imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = self.imageSW.scrambleImage(level-1)
            imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = self.imageNW.scrambleImage(level-1)
            imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = self.imageNE.scrambleImage(level-1)
        elif(level%2==1):
            imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = self.imageNW.scrambleImage(level-1)
            imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = self.imageNE.scrambleImage(level-1)
            imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = self.imageSE.scrambleImage(level-1)
            imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = self.imageSW.scrambleImage(level-1)

        return imageResult

def imageUnion(image1, image2, level):
    resultHeight = image1.height
    resultWidth = image2.width
    imageResult = np.zeros((resultHeight, resultWidth, 3), np.uint8)

    if(level==0):
        colors = (image1.mean, image2.mean)
        averageColor = tuple(map(np.mean, zip(*colors)))
        return averageColor 
    else:
        imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = imageUnion(image1.imageNW, image2.imageNW, level-1)
        imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = imageUnion(image1.imageNE, image2.imageNE, level-1)
        imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = imageUnion(image1.imageSE, image2.imageSE, level-1)
        imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = imageUnion(image1.imageSW, image2.imageSW, level-1)
        
    return imageResult

def showImage(imageName):
    cv2.imshow("Lenna",imageName)
    cv2.waitKey(0)

