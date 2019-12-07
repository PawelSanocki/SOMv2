# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# import os
# g_login = GoogleAuth()
# g_login.LocalWebserverAuth()
# drive = GoogleDrive(g_login)
# with open("results\\dc\\2019-10-31 13_44_57-.png","r") as file:
#     file_drive = drive.CreateFile({'title':os.path.basename(file.name) })  
#     file_drive.SetContentString(file.read()) 
#     file1_drive.Upload()
import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
#a = np.transpose(a)
np.random.shuffle(a)
print(a)