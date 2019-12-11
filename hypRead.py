import spectral.io.envi as envi
import numpy as np
import MySOM
import cv2
from spectral import open_image
from os.path import join
from scipy.io import loadmat
import time
import nmi

def segmentImage(folderPath, outputPath,  imageFile, n_iter, learn_rate, threshold, dx = 16, dy = 16, dz = 16, showResult = False, size = 1, output_quality = 1, sig = None):
    if imageFile.find(".lan") > 0:
        filename = imageFile.replace(".lan","")
        img = open_image(folderPath + "\\" + imageFile)
    elif imageFile.find(".hdr") > 0:
        filename = imageFile.replace(".hdr","")
        img = envi.open(folderPath + "\\" + imageFile)
    elif imageFile.find(".mat") > 0:
        filename = imageFile.replace(".mat","")
        img_mat = loadmat(folderPath + "\\" + imageFile)
        for i in img_mat:
            img = img_mat[i]
    elif imageFile.find(".png") > 0 or imageFile.find(".jpg") > 0:
        filename = imageFile.replace(".jpg","")
        filename = filename.replace(".png","")
        img = cv2.imread(folderPath + "\\" + imageFile)
    else:
        print("Wrong input")
        return
    print(img.shape[0]*img.shape[1])
    my_som = MySOM.Som(dim_x = dx,dim_y = dy,dim_z = dz,input_dim = img.shape[2],learning_rate = learn_rate, learn_iter = n_iter, size=size, quality = output_quality, sigma=sig)
    #my_som.load_weights(filename)
    #my_som.train_with_threshold_hyperspectral(threshold, folderPath)
    start_time = time.time()
    my_som.train(threshold*(size**2), folderPath)
    print("Time taken for training: " + str((time.time() - start_time)/60) + " min")
    #my_som.save_weights(filename)
    start_time = time.time()
    result = my_som.convert_image(img)
    print("Time taken for converting: " + str((time.time() - start_time)/60) + " min")
    outputFileName = outputPath + "\\" + filename + "\\" + str(time.time()//1) + str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations) + "_th" + str(threshold) + "_" + "_s" + str(size) + ".jpg"
    print(outputFileName)
    print(cv2.imwrite(outputFileName,result))
    my_som.finish()
    nmi.validateResult(imgFile = outputFileName, referencePath= "D:\\Results\\results\\dc.png")
    if showResult == True:
        cv2.imshow(str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations), result)
        cv2.waitKey()

