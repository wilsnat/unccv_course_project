from PIL import  Image
import piexif
import os
#import StringIO

list = os.listdir('dataset_full')


#for i,file in enumerate(list):
for i in range(000, 320):
	file_path = 'dataset_full/'+str(i).zfill(4)+'.JPG'

	im = Image.open(file_path)

	# load exif data
	exif_dict = piexif.load(im.info["exif"])
	exif_bytes = piexif.dump(exif_dict)

	THUMB_SIZES = [(250, 250)]
	for thumbnail_size in THUMB_SIZES:
	    im.thumbnail( thumbnail_size, Image.ANTIALIAS)
	    #thumbnail_buf_string = StringIO.StringIO()
	    # save thumbnail with exif data
	    im.save( 'resized_img/'+str(i).zfill(4)+ ".jpg", "JPEG", exif=exif_bytes)