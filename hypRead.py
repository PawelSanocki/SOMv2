import spectral.io.envi as envi
import numpy as np
import MySOM
import cv2
from spectral import open_image
from os.path import join
from scipy.io import loadmat
import time
import nmi

def segmentImage(folderPath, outputPath,refPath,  imageFile, n_iter, learn_rate, threshold, dx = 16, dy = 16, dz = 16, showResult = False, size = 1, output_quality = 1, sig = None):
    '''
    Method created for experimental validation of the program.
    folderPath - folder with hyperspectral image to be converted and other imeges which will take part in learning process.
    outputPath - output path of the program.
    refPath - path to groundtruth image.
    imageFile - name of the hyperspectral image to be converted.
    n_iter - number of learning iterations.
    learn_rate - initaial learning rate of the algorithm.
    threshold - threshold value, used when creating the training set.
    dx - size of first dimension of SOM.
    dx - size of second dimension of SOM.
    dz - size of third dimension of SOM.
    showResult - boolean value, printing the result of conversion on the screen.
    size - size of the neighbourhood.
    output_quality - in convertion.
    sig - initial size of neighbourhood.
    '''
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
    
    my_som = MySOM.Som(dim_x = dx,dim_y = dy,dim_z = dz,input_dim = img.shape[2],learning_rate = learn_rate, learn_iter = n_iter, size=size, quality = output_quality, sigma=sig)
    
    start_time = time.time()
    my_som.train(threshold*(size**2), folderPath)
    print("Time taken for training: " + str((time.time() - start_time)/60) + " min")
    
    start_time = time.time()
    result = my_som.convert_image(img)
    print("Time taken for converting: " + str((time.time() - start_time)/60) + " min")
    outputFileName = outputPath + "\\" + filename + "\\" + str(time.time()//1) + str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations) + "_th" + str(threshold) + "_" + "_s" + str(size) + ".png"
    print(outputFileName)
    print(cv2.imwrite(outputFileName,result))
    my_som.finish()
    nmi.validateResult(imgPath = outputFileName, referencePath = refPath, img = result)
    if showResult == True:
        cv2.imshow(str(my_som.d_x) + str(my_som.d_y) + str(my_som.d_z) + "_lr" + str(int(1//my_som.lr)) + "_li" + str(my_som._learn_iterations), result)
        cv2.waitKey()

