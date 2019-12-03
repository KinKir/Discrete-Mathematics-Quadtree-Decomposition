import cv2
import numpy as np
import os
import shutil

class quadtree: 
    def __init__(self, image, height, width):

        # Set mean and STD of image region
        self.mean, self.std = cv2.meanStdDev(image)
        self.mean = tuple(self.mean.flatten())
        self.std = sum(x for x in self.std.flatten()) / self.std.size

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

# Procedure for segementing a quadtree image    
def quadtreeSegmentation(filename, limit=7, stdLimit=10.0, write=False):
    imagePath = "./images/" + filename
    imageName = filename.split('.')[0]
    resultPath = "./result/"

    imageInput = cv2.imread(imagePath)

    print("Building quadtree representation of image..")
    Q = quadtree(imageInput, imageInput.shape[0], imageInput.shape[1])

    # Define queue for BFS
    q = []
    coordinateQueue1 = []
    coordinateQueue2 = []
    levelQueue = []

    q.append(Q)
    coordinateQueue1.append((0, 0))
    coordinateQueue2.append((imageInput.shape[0], imageInput.shape[1]))
    levelQueue.append(1)

    # Define result size
    resultHeight = imageInput.shape[0]
    resultWidth = imageInput.shape[1]
    imageResult = np.zeros((resultHeight,resultWidth,3), np.uint8)

    cv2.imshow("Lenna", imageResult)
    cv2.waitKey(0)

    while(len(q)!=0):
        # Pop front of all queues
        currentNode = q.pop(0)
        y1, x1 = coordinateQueue1.pop(0)
        y2, x2 = coordinateQueue2.pop(0)
        currentLevel = levelQueue.pop(0)

        # Find midpoint
        halfX = (x1+x2)//2
        halfY = (y1+y2)//2

        # If the STD is still larger than the stdLimit...
        if(currentNode.std>=stdLimit and currentLevel<=limit):
            # Partition NW region
            imageResult[y1:halfY, x1:halfX] = currentNode.imageNW.mean
            q.append(currentNode.imageNW)
            coordinateQueue1.append((y1,x1))
            coordinateQueue2.append((halfY,halfX))
            levelQueue.append(currentLevel+1)

            # Partition NE region
            imageResult[y1:halfY, halfX:x2] = currentNode.imageNE.mean
            q.append(currentNode.imageNE)
            coordinateQueue1.append((y1,halfX))
            coordinateQueue2.append((halfY,x2))
            levelQueue.append(currentLevel+1)

            # Partition SE region
            imageResult[halfY:y2, halfX:x2] = currentNode.imageSE.mean
            q.append(currentNode.imageSE)
            coordinateQueue1.append((halfY,halfX))
            coordinateQueue2.append((y2,x2))
            levelQueue.append(currentLevel+1)
            
            # Partition SW region
            imageResult[halfY:y2, x1:halfX] = currentNode.imageSW.mean
            q.append(currentNode.imageSW)
            coordinateQueue1.append((halfY,x1))
            coordinateQueue2.append((y2,halfX))
            levelQueue.append(currentLevel+1)
            
            showImage(imageResult)
        # else:
        #     imageResult[y1:y2, x1:x2] = currentNode.mean
    

    print("Segmentation complete!")
    cv2.waitKey(0)
    cv2.imwrite(resultPath + "lenna_modified.png", imageResult)


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

