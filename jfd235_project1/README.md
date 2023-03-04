To select regions of interest from images:
Run train.py to do ROI poly. Run once for each data class (this has already been done)

To train model based on image data:
Run generateTable.py with the directories pointing to the directories pointing to the outputs of the previous step (this has already been done)

To test model on new images:
Run train2.py with img_dir as the directory containing the images you want to test, and out_dir as the directory you want the images to be saved to