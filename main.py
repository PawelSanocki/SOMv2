import hypRead
op = "D:\\Results\\results"
fp = "D:\\Results\\h_data"

#im = "dc.lan" # dc threshold = 11000 naive
#rp = "D:\\Results\\results\\dc.png"

#im = "Salinas.mat" # threshold = 5000 naive, 3500 for 3, 2200 for 5, rot 3 -> th = 3400, rot 5 -> th = 2400, rot 7 -> 1400
#rp = "D:\\Results\\results\\Salinas.png"

# PaviaU 103 spectral bands
im = "PaviaU.mat" #th = 12k naive, th = 8000 rotational s = 3, th = 4000 s = 5
rp = "D:\\Results\\results\\PaviaU.png"


hypRead.segmentImage(folderPath=fp, outputPath=op, refPath = rp, imageFile = im, learn_rate=0.1, n_iter=1000, \
                    threshold = 12000, dx = 6, dy = 6, dz = 6, size=1, output_quality=1, showResult=False, sig = None)


