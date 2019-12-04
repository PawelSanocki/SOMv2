from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import cv2
from scipy.io import loadmat
import numpy as np

def generator_image(img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                yield i, j

def createVector(img):
    result = np.empty(0)
    for i in generator_image(img):
        #print(img[i][0])
        result = np.append(result, img[i][0]*256*256 + img[i][1]*256 + img[i][2])
    return result


referencePath = "D:\\Results\\results\\PaviaUref.png"
folderPath = "D:\\Results\\results\\PaviaU\\"
imgFile = "101010-lr9-li10-1572729738.867069.jpg"

ref = cv2.imread(referencePath)

img = cv2.imread(folderPath + imgFile)
img = cv2.resize(img, (ref.shape[1], ref.shape[0]))

cv2.imshow("img", img)
cv2.imshow("ref", ref)
cv2.waitKey()
cv2.destroyAllWindows()
ref = createVector(ref)
print("converting done")
img = createVector(img)
print("converting done")
print("NMI: " + str(normalized_mutual_info_score(ref, img)))
print("ARS: " + str(adjusted_rand_score(ref, img)))


