import hypRead
op = "D:\\Results\\results"
fp = "D:\\Results\\h_data"

#im = "dc.lan" # dc threshold = 11000 naive
#rp = "D:\\Results\\results\\dc.png"

#im = "Salinas.mat" # threshold = 5000 naive
#rp = "D:\\Results\\results\\Salinas.png"

# PaviaU 103 spectral bands
im = "PaviaU.mat" #th = 14000 naive, th = 8000 rotational
rp = "D:\\Results\\results\\PaviaU.png"


#im = "SalinasA.mat"
#im = "Indian_pines.mat"
#im = "a.jpg" # th = 100


hypRead.segmentImage(folderPath=fp, outputPath=op, refPath = rp, imageFile = im, learn_rate=0.1, n_iter=1000, \
                    threshold = 8000, dx = 6, dy = 6, dz = 6, size=3, output_quality=3, showResult=False, sig = None)

# analiza problemu
# zapoznanie z algorytmai i przegląd literatury
# moja metoda opis itd
# wyniki
# podsumowanie

