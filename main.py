import quadtree

imageName = "liechestein"
imageName2 = "airplane"
folderPath = "./result/"+imageName

imageInput = cv2.imread("./images/" + imageName + ".png")
imageInput2 = cv2.imread("./images/" + imageName2 + ".png")

height, width = imageInput.shape[0], imageInput.shape[1]

print("Building quadtree representation of image...")
Q = quadtree(imageInput, height, width)
Q2 = quadtree(imageInput2, height, width)

showImage(imageUnion(Q,Q2,8))
showImage(Q.scrambleImage(8))

print("Building quadtree representation of image...")
Q3 = quadtree(Q.scrambleImage(8), height,width)
showImage(Q3.scrambleImage(8))

# Q.printTree(0)
# os.mkdir(folderPath)

# for i in range(1,10):
#     image = Q.createLeveledCompression(Q, i)
#     # showImage(image)
#     cv2.imwrite(folderPath + "/" + str(1<<i) + "x" + str(1<<i) + ".png", image)
#     print("Converted image of size " + str(1<<i) + "x" + str(1<<i))

