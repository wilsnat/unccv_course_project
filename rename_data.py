import os

folder = "./dataset_full/set01"

list = os.listdir(folder)
os.chdir(folder)
for idx, file in enumerate(list):
    if(str(file)[-4:]==".jpg"):
        os.rename(file, str(idx+160).zfill(4) + ".jpg")
