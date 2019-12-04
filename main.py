import hypRead
import time
op = "D:\\Results\\results"
fp = "C:\\Users\\Paweł\\Desktop\\h_data"
#im = "dc.lan"
#im = "Salinas.mat"
im = "PaviaU.mat"
# PaviaU threshold = 260000
#im = "SalinasA.mat"
#im = "Indian_pines.mat"

hypRead.segmentImage(folderPath=fp, outputPath=op, imageFile = im, learn_rate=0.01, n_iter=1000, \
                    threshold = 265000, dx = 8, dy = 8, dz = 8, size=1, output_quality=1, showResult=False)

# analiza problemu
# zapoznanie z algorytmai i przegląd literatury
# moja metoda opis itd
# wyniki
# podsumowanie

