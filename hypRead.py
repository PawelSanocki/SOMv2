import spectral.io.envi as envi
import numpy as np
import MySOM
import cv2
from spectral import open_image
from os.path import join
from scipy.io import loadmat
import time

def segmentImage(folderPath, outputPath,  imageFile, n_iter, learn_rate, threshold, dx = 16, dy = 16, dz = 16, showResult = False, size = 1, output_quality = 1):
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
    else:
        print("Wrong input")
        return
    my_som = MySOM.Som(dim_x = dx,dim_y = dy,dim_z = dz,input_dim = img.shape[2],learning_rate = learn_rate, learn_iter = n_iter, size=size, quality = output_quality)
    #my_som.load_weights()
    #my_som.train_with_threshold_hyperspectral(threshold, folderPath)
    my_som.train(threshold*(size**2), folderPath)
    #my_som.save_weights()
    result = my_som.convert_image(img)
    print(cv2.imwrite(outputPath + "\\" + filename + "\\" + str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations) + "_th" + str(threshold) + "_" + str(time.time()) + "_s" + str(size) + ".jpg",result))
    my_som.finish()
    if showResult == True:
        cv2.imshow(str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations), result)
        cv2.waitKey()

