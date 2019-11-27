import os
 
path = "C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data"
#print ("Pliki w folderze: ")
filenames = os.listdir(path)
#print (filenames)
i = 0
for filename in filenames:
    os.rename("C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data\\" + filename,"C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data\\" + str(i) + ".jpg")
    i += 1