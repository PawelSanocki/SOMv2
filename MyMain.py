import MySOM as som
import cv2
import numpy as np


my_som = som.Som(dim_x = 8,dim_y = 8,dim_z = 8,input_dim = 3,learning_rate = 0.001, learn_iter = 200)
#my_som.load_weights()
#my_som.train_image()
my_som.train_with_threshold(20)
print("training done")
my_som.save_weights()
print("saved")
img = cv2.imread("C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data\\0.jpg")
img1 = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
cv2.imshow("converted", cv2.resize(my_som.convert_image(img1), (600,600)))
cv2.imshow("base", cv2.resize(img, (600,600)))
print("converted")

cv2.waitKey()
