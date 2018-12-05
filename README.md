# unccv_course_project
#### Devansh Desai | Shamika Kulkarni | Shweta Patil | Nate Wilson

Appendix:

**dataset_full:** folder with data

  **Set01:** set of 320 images of 32 colors
  
  **exif01:** metadata from images of set01, only the exposure time and iso is consistant
  

  **lab_colors01:** colors for set01 from colorreader, in Lab, the first row is the corresponding number of images per color
  
  **Set02:** set of 320 images of 32 colors
  
  **exif02:** metadata from images of set01, only the exposure time and iso is consistant
  
  **lab_colors02:** colors for set01 from colorreader, in Lab, the first row is the corresponding number of images per color
  
  **train_y: **

**project_documents:** all the documents from the project
  
  **Group10_Final_Project.pdf:** final presentation
  
  **Vishion Literature Review.pdf:** literature review of related research
  
  **Vishion Market Research.pdf:** exploration of our customer and their market
  
  **Vishion Technical Plan.pdf:** how we planned this repository

**slideshow_images:** images from the slideshow

**README.md:** You're reading it!

**classification_data_loader.py:** for nates_classification, imports the data a bit differently with 32 classfications instead of color values

**color_extraction.py:**	k means algorithm for extracting dominant color and color name detector

**connected_component.py:**	grab cut mask shape changed

**data_loader.py:**	loads data for nates_regression and nates_regression_redux, imports the data with raw colors

**data_visualization.py:** creates chart of color classifications (used in presentation)

**nates_classification.py:** classifcation algorithm

**nates_regression.py:** simple regression network

**nates_regression_redux.py:** more complex regression network

**rename_data.py:** dataset prep (renames files)

**resize_data.py:** dataset prep (resized image)

**shweta_kmeans_color_extractor.py:**	k means clustering code changes	17 hours ago

**simple_color_detector.py:**	k means algorithm for extracting dominant color and color name detector	a day ago

**strip_data.py:** dataset prep (copies metadata to exif0*)
