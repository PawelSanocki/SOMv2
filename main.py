import hypRead
op = "D:\\Results\\results"
fp = "D:\\Results\\h_data"
im = "dc.lan" # dc threshold = 11000
#im = "Salinas.mat" # threshold = 5000
#im = "PaviaU.mat" #th = 14000
#im = "SalinasA.mat"
#im = "Indian_pines.mat"
#im = "a.jpg" # th = 100

hypRead.segmentImage(folderPath=fp, outputPath=op, imageFile = im, learn_rate=0.1, n_iter=1000, \
                    threshold = 11000, dx = 16, dy = 16, dz = 16, size=3, output_quality=1, showResult=False, sig = None)

# analiza problemu
# zapoznanie z algorytmai i przegląd literatury
# moja metoda opis itd
# wyniki
# podsumowanie

