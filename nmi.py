from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import cv2
from scipy.io import loadmat
import numpy as np
def validateResult(imgPath = None, referencePath = None, img = None, ref = None):
    '''
    Method for validating the result using the NMI index.
    imgPath - path of the converted image.
    reference-path - path of the groundtruth file.
    img - numpy array with image, if no path specified.
    ref - numpy array with groundtruth, if no path specified.
    '''
    def generator_image(img):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    yield i, j

    def isPresent(vec, lib):
        for i in range(lib.shape[1]):
            if vec[0] == lib[0][i] and vec[1] == lib[1][i] and vec[2] == lib[2][i]:
                return True
        print(lib.shape)
        print(vec.shape)
        return False

    def createVector(img, ref):
        image = np.empty(0)
        reference = np.empty(0)
        lib = np.empty((3,0))
        for i in generator_image(ref):
            # if lib.shape[1] == 0:
            #     lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
            # elif (isPresent(img[i], lib)) is False:
            #     lib = np.append(lib, np.resize(img[i], (3,1)), axis=1)
            if (ref[i][0] == 0 and ref[i][1] == 0 and ref[i][2] == 0):
                continue
            image = np.append(image, img[i][0]*256*256 + img[i][1]*256 + img[i][2])
            reference = np.append(reference, ref[i][0]*256*256 + ref[i][1]*256 + ref[i][2])
        #print(lib.shape)
        return image, reference,  lib.shape[1]
    if not isinstance(ref, np.ndarray):
        ref = cv2.imread(referencePath)
    if not isinstance(img, np.ndarray):
        img = cv2.imread(imgPath)
    img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    
    [img, ref, n_colors] = createVector(img, ref)
    print("NMI: " + str(normalized_mutual_info_score(ref, img)))
    #print("ARS: " + str(adjusted_rand_score(ref, img)))
    #print("Number of colors: " + str(n_colors))
    # cv2.destroyAllWindows()
    # cv2.imshow("img", img)
    # cv2.imshow("ref", ref)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

#validateResult(r"D:\Results\results\Salinas\1578735550.0161616_lr99_li1000_th5000__s1.jpg", r"D:\Results\results\Salinas.png")