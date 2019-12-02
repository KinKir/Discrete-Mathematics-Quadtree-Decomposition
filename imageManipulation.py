import quadtree
import numpy as np

class imageManipulation:
    def __init__(self, level):
        
        # Initialize result size
        self.imageResult = np.zeros((1<<level, 1<<level, 3), np.uint8)

    # Method for setting up quadrants
    def setQuadrants(self, height, width, NW, NE, SE, SW):
        self.imageResult[0:(height>>1), 0:(width>>1)] = NW
        self.imageResult[0:(height>>1), (width>>1):width] = NE
        self.imageResult[(height>>1):height, (width>>1):width] = SE
        self.imageResult[(height>>1):height, 0:(width>>1)] = SW

    # Method for leveled scaling
    def createLeveledScaling(self, image, level):
        h, w = image.height, image.width

        if(level==0):
            return image.mean
        else:
            self.setQuadrants(
                h, 
                w, 
                self.createLeveledScaling(image.imageNW, level-1),
                self.createLeveledScaling(image.imageNE, level-1),
                self.createLeveledScaling(image.imageSE, level-1),
                self.createLeveledScaling(image.imageSW, level-1),
            )

        