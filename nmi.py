from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import cv2
from scipy.io import loadmat
import numpy as np
def validateResult(imgFile = "D:\\Results\\results\\PaviaU\\1575712757.065766161616_lr999_li1000_th300000__s1.jpg", referencePath = "D:\\Results\\results\\PaviaU.png"):
    def generator_image(img):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    yield i, j

    def createVector(img, ref):
        image = np.empty(0)
        reference = np.empty(0)
        lib = np.empty((3,0))
        for i in generator_image(ref):
            if lib.size == 0:
                lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
            elif (np.resize(img[i], (3,1)) in lib) is False:
                lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
            if (ref[i][0] == 0 and ref[i][1] == 0 and ref[i][2] == 0):
                continue
            image = np.append(image, img[i][0]*256*256 + img[i][1]*256 + img[i][2])
            reference = np.append(reference, ref[i][0]*256*256 + ref[i][1]*256 + ref[i][2])
        #print(str(image.shape) + "_" + str(reference.shape))
        return image, reference,  lib.shape[1]
        
    ref = cv2.imread(referencePath)

    img = cv2.imread(imgFile)
    img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    
    [img, ref, n_colors] = createVector(img, ref)
    print("converting done")
    print("NMI: " + str(normalized_mutual_info_score(ref, img)))
    print("ARS: " + str(adjusted_rand_score(ref, img)))
    print("Number of colors: " + str(n_colors))
    # cv2.destroyAllWindows()
    # cv2.imshow("img", img)
    # cv2.imshow("ref", ref)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

#validateResult()