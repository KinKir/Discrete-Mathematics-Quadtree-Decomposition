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
        print('HW:', self.height,self.width)
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


def showImage(imageName):
    cv2.imshow("Lenna",imageName)
    cv2.waitKey(0)

imageName = "lenna"
folderPath = "./result/"+imageName

imageInput = cv2.imread("./images/" + imageName + ".png")

height, width = imageInput.shape[0], imageInput.shape[1]

Q = quadtree(imageInput, height, width)

os.mkdir(folderPath)

for i in range(1,10):
    image = Q.createLeveledCompression(Q, i)
    # showImage(image)
    cv2.imwrite(folderPath + "/" + str(1<<i) + "x" + str(1<<i) + ".png", image)
    print("Converted image of size " + str(1<<i) + "x" + str(1<<i))
