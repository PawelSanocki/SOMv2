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
    lib = np.empty((3,0))
    for i in generator_image(img):
        #print(lib.shape)
        result = np.append(result, img[i][0]*256*256 + img[i][1]*256 + img[i][2])
        if lib.size == 0:
            lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
        elif (np.resize(img[i], (3,1)) in lib) is False:
            lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
            
    #print(lib.shape)
    return result, lib.shape[1]


referencePath = "D:\\Results\\results\\PaviaU.png"
folderPath = "D:\\Results\\results\\PaviaU\\"
imgFile = "1575469144.8332124121212_lr9_li150_th80000__s1.jpg"

ref = cv2.imread(referencePath)

img = cv2.imread(folderPath + imgFile)
img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
cv2.imshow("img", img)
cv2.imshow("ref", ref)
cv2.waitKey()
cv2.destroyAllWindows()
[img, n_colors] = createVector(img)
print("converting done")
print("Number of colors: " + str(n_colors))
ref, not_important = createVector(ref)
print("Number of not important: " + str(not_important))
print("converting done")
print("NMI: " + str(normalized_mutual_info_score(ref, img)))
print("ARS: " + str(adjusted_rand_score(ref, img)))
print("Number of colors: " + str(n_colors))


