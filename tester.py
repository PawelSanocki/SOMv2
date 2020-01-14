import cv2
import numpy as np
import time

img = cv2.imread(r"D:\Results\results\dct.png")
for i in range(2):
    img  = cv2.bilateralFilter(img,9,50,50)
    img  = cv2.medianBlur(img, 3)

name = "D:\\Results\\results\\ref\\" + "dc" + ".jpg"
print(name)
print(cv2.imwrite(name,img))

def isPresent(vec, lib):
    for i in range(lib.shape[0]):
        if vec[0] == lib[i][0] and vec[1] == lib[i][1] and vec[2] == lib[i][2]:
            return True
    return False

a = np.array([[1,2,3], [2,3, 4], [3,4, 5], [11,11,11]])
print(a.shape)
v = np.array([11, 11, 11])
print(v.shape)
print(isPresent(v,a))