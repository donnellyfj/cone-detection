# Cone Detection
This project is a model that is able to detect traffic cones in an image, and will create a bounding box around the cone, as well as report how far away from the camera the cone is. Instructions can be found below, and details on the implementation and test results can be found in the Report PDF file.

## Instructions
To select regions of interest from images:
Run train.py to do ROI poly. Run once for each data class (this has already been done)

To train model based on image data:
Run generateTable.py with the directories pointing to outputs of the previous step (this has already been done)

To test model on new images:
Run train2.py with img_dir as the directory containing the images you want to test, and out_dir as the directory you want the images to be saved to