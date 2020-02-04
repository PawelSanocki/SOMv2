import hypRead
'''
op - output path, destination in which result of the program will be stored
fp - folder with hyperspectral images
im - name of file with hyperspectral image
rp - reference path, path of groundtruth image

lr - learning rate
li - number of learning iterations
th - threshold for building training set
dx - size of grid in first dimension
dy - size of grid in second dimension
dz - size of grid in third dimension
s - size of neighbourhood for building input vector process
q - quality
sigma - initial size of neighbourhood for training process
showResult - printing the result of segmentation of the screen
'''
op = "D:\\Results\\results"
fp = "D:\\Results\\h_data"
im = "PaviaU.mat" 
rp = "D:\\Results\\results\\PaviaU.png"

lr = 0.1
li = 1000
th = 12000
dx = 6
dy = 6
dz = 6
s = 1
q = 5
sigma = None
showResult = False

#im = "Salinas.mat"
#rp = "D:\\Results\\results\\Salinas.png"


hypRead.segmentImage(folderPath=fp, outputPath=op, refPath = rp, imageFile = im, learn_rate=lr, n_iter=li, \
                    threshold = th, dx = dx, dy = dy, dz = dz, size=s, output_quality=q, showResult=showResult, sig = sigma)


