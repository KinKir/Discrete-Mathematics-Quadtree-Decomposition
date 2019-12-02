import quadtree as q
import cv2

# imageName = "liechestein"
# imageInput = cv2.imread("./images/" + imageName + ".png")

# height, width = imageInput.shape[0], imageInput.shape[1]

# print("Building quadtree representation of image...")
# Q = q.quadtree(imageInput, height, width)

q.exportScaling("lenna.png")

# for i in range(1,10):
#     result = q.createLeveledScaling(Q, i)
#     showImage(result)

# imageName2 = "airplane"
# folderPath = "./result/"+imageName

# imageInput2 = cv2.imread("./images/" + imageName2 + ".png")


# Q = quadtree(imageInput, height, width)
# Q2 = quadtree(imageInput2, height, width)

# showImage(imageUnion(Q,Q2,8))
# showImage(Q.scrambleImage(8))

# print("Building quadtree representation of image...")
# Q3 = quadtree(Q.scrambleImage(8), height,width)
# showImage(Q3.scrambleImage(8))

# # Q.printTree(0)
# # os.mkdir(folderPath)

# # for i in range(1,10):
# #     image = Q.createLeveledCompression(Q, i)
# #     # showImage(image)
# #     cv2.imwrite(folderPath + "/" + str(1<<i) + "x" + str(1<<i) + ".png", image)
# #     print("Converted image of size " + str(1<<i) + "x" + str(1<<i))

