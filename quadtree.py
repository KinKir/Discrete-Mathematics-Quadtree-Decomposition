import cv2
import numpy as np
import os
import shutil
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

        # Build the tree (top-down)
        if(height!=1 and width!=1):
            self.imageNW = quadtree(image[0:self.halfHeight, 0:self.halfWidth], self.halfHeight, self.halfWidth)
            self.imageNE = quadtree(image[0:self.halfHeight, self.halfWidth:width], self.halfHeight, self.halfWidth)
            self.imageSE = quadtree(image[self.halfHeight:height, self.halfWidth:width], self.halfHeight, self.halfWidth)
            self.imageSW = quadtree(image[self.halfHeight:height, 0:self.halfWidth], self.halfHeight, self.halfWidth)

    # Method for debugging purposes
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

# --------------------------------------

# Procedure for setting quadrants
def setQuadrants(imageResult, NW, NE, SE, SW):
    h, w = imageResult.shape[0], imageResult.shape[1]

    imageResult[0:(h>>1), 0:(w>>1)] = NW
    imageResult[0:(h>>1), (w>>1):w] = NE
    imageResult[(h>>1):h, (w>>1):w] = SE
    imageResult[(h>>1):h, 0:(w>>1)] = SW

    return imageResult

# Procedure for leveled scaling
def createLeveledScaling(image, level):
    resultHeight = 1<<level
    resultWidth = 1<<level
    imageResult = np.zeros((resultHeight,resultWidth,3), np.uint8)

    if(level==0):
        return image.mean
    else:
        return setQuadrants(
            imageResult,
            createLeveledScaling(image.imageNW, level-1),
            createLeveledScaling(image.imageNE, level-1),
            createLeveledScaling(image.imageSE, level-1),
            createLeveledScaling(image.imageSW, level-1),
        )

# Procedure for exporting to folder
def exportScaling(filename, show=False):

    imagePath = "./images/" + filename
    imageName = filename.split('.')[0]

    resultPath = "./result/" + imageName
    
    imageInput = cv2.imread(imagePath)
    Q = quadtree(imageInput, imageInput.shape[0], imageInput.shape[1])

    print("Creating folder" + resultPath + "...")
    
    # Making an output folder
    try:
        os.mkdir(resultPath)
    except OSError:
        ans = input("Folder already exists, delete folder? [Y/N] : ")
        if(ans=='y' or ans=='Y'):
            shutil.rmtree(resultPath)
            os.mkdir(resultPath)
        else:
            raise Exception("Directory already exists. Remove or move it somewhere else.")

    for i in range(0,10):
        size = 1<<i
        result = createLeveledScaling(Q, i)
        if(show):
            showImage(result)
        print("Writing image of size " + str(size) + "x" + str(size) + "...")
        cv2.imwrite(resultPath + "/" + str(size) + "x" + str(size) + ".png", result)
        
            



#     def scrambleImage(self, level):
#         resultHeight = self.height
#         resultWidth = self.width
#         imageResult = np.zeros((resultHeight, resultWidth, 3), np.uint8)

#         if(level==0):
#             return self.mean
#         elif(level%2==0):
#             imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = self.imageSE.scrambleImage(level-1)
#             imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = self.imageSW.scrambleImage(level-1)
#             imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = self.imageNW.scrambleImage(level-1)
#             imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = self.imageNE.scrambleImage(level-1)
#         elif(level%2==1):
#             imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = self.imageNW.scrambleImage(level-1)
#             imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = self.imageNE.scrambleImage(level-1)
#             imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = self.imageSE.scrambleImage(level-1)
#             imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = self.imageSW.scrambleImage(level-1)

#         return imageResult

# def imageUnion(image1, image2, level):
#     resultHeight = image1.height
#     resultWidth = image2.width
#     imageResult = np.zeros((resultHeight, resultWidth, 3), np.uint8)

#     if(level==0):
#         colors = (image1.mean, image2.mean)
#         averageColor = tuple(map(np.mean, zip(*colors)))
#         return averageColor 
#     else:
#         imageResult[0:(resultHeight>>1), 0:(resultWidth>>1)] = imageUnion(image1.imageNW, image2.imageNW, level-1)
#         imageResult[0:(resultHeight>>1), (resultWidth>>1):resultWidth] = imageUnion(image1.imageNE, image2.imageNE, level-1)
#         imageResult[(resultHeight>>1):resultHeight, (resultWidth>>1):resultWidth] = imageUnion(image1.imageSE, image2.imageSE, level-1)
#         imageResult[(resultHeight>>1):resultHeight, 0:(resultWidth>>1)] = imageUnion(image1.imageSW, image2.imageSW, level-1)
        
#     return imageResult

def showImage(imageName):
    cv2.imshow("Lenna",imageName)
    cv2.waitKey(0)

