import hypRead
import time
op = "D:\\Results\\results"
fp = "C:\\Users\\Paweł\\Desktop\\h_data"
im = "dc.lan"
#im = "Salinas.mat"
#im = "PaviaU.mat"
#im = "SalinasA.mat"
#im = "Indian_pines.mat"
start_time = time.time()
hypRead.segmentImage(folderPath=fp, outputPath=op, imageFile = im, learn_rate=0.1, n_iter=150, \
                    threshold = 1800, dx = 16, dy = 16, dz = 16, size=3, output_quality=1, showResult=False)
print("Time taken: " + str((time.time() - start_time)/60) + " min")
# analiza problemu
# zapoznanie z algorytmai i przegląd literatury
# moja metoda opis itd
# wyniki
# podsumowanie
