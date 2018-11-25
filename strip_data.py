import exifread as exif
import os, csv

folder = "./test/"

list = os.listdir(folder)
os.chdir(folder)
tag_keys = ['EXIF ExposureTime', 'EXIF FNumber', 'EXIF ExposureProgram', 'EXIF ISOSpeedRatings', 'EXIF ExifVersion', 'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'EXIF ComponentsConfiguration', 'EXIF ShutterSpeedValue', 'EXIF ApertureValue']
with open('exif.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=tag_keys)

    writer.writeheader()

    for idx, file in enumerate(list):
        if(str(file)[-4:]==".jpg" or str(file)[-4:]==".JPG"):
            f = open(file, 'rb')
            tags = exif.process_file(f)
            exif_tags = [tags[x] for x in tag_keys]
            writer.writerow(dict(zip(tag_keys, exif_tags)))
