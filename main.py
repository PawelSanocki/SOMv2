import hypRead
import time
op = "D:\\Results\\results"
fp = "C:\\Users\\Paweł\\Desktop\\h_data\\Salinas"
im = "dc.lan"
# dc threshold = 1750000
im = "Salinas.mat"
# Salinas threshold = 630000
#im = "PaviaU.mat"
# PaviaU threshold = 300000
#im = "SalinasA.mat"
#im = "Indian_pines.mat"

hypRead.segmentImage(folderPath=fp, outputPath=op, imageFile = im, learn_rate=0.01, n_iter=10000, \
                    threshold = 5000, dx = 12, dy = 12, dz = 12, size=1, output_quality=1, showResult=False, sig = None)

# analiza problemu
# zapoznanie z algorytmai i przegląd literatury
# moja metoda opis itd
# wyniki
# podsumowanie

