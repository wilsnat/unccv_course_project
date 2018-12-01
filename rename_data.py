import os

folder = "./dataset_full/set02"

list = os.listdir(folder)
os.chdir(folder)
for idx, file in enumerate(list):
    if(str(file)[-4:]==".jpg"):
        os.rename(file, str(idx).zfill(4) + ".jpg")
