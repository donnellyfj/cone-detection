import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
from roipoly import RoiPoly

img_dir = "ECE5242Proj1-trainRed/"
out_dir = "train-outRed/"
for filename in os.listdir(img_dir):
    if filename.endswith(".png"):
        image = plt.imread(img_dir + filename)
        outname = filename[0:-3] + "csv"

        # Show the image
        fig = plt.figure()
        plt.imshow(image, interpolation='nearest', cmap="Greys")
        plt.colorbar()
        plt.title("left click: line segment         right click or double click: close region")
        plt.show(block=False)

        # Let user draw first ROI
        roi1 = RoiPoly(color='r', fig=fig)

        # Show the image with the first ROI
        # fig = plt.figure()
        # plt.imshow(image, interpolation='nearest', cmap="Greys")
        # plt.colorbar()
        # roi1.display_roi()
        # plt.title('draw second ROI')
        # plt.show(block=False)

        # Let user draw second ROI
        # roi2 = RoiPoly(color='b', fig=fig)

        # Show the image with both ROIs and their mean values
        # plt.imshow(image, interpolation='nearest', cmap="Greys")
        # plt.colorbar()

        # roi1.display_roi()
        # roi1.display_mean(image[:,:,0])
        # plt.title('The two ROIs')
        # plt.show()

        # Show ROI masks
        mask1 = roi1.get_mask(image[:,:,0])
        # mask2 = ~roi2.get_mask(image[:,:,0])
        # plt.imshow(mask1 + mask2,
        #            interpolation='nearest', cmap="Greys")
        # print(image[mask1])
        # plt.imshow(mask2)
        # print(mask2)
        # plt.title('ROI masks of the two ROIs')
        # plt.show()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.scatter3D(image[mask1][:,0], image[mask1][:,1], image[mask1][:,2], c=image[mask1])
        # ax.scatter3D(image[mask2][:,0], image[mask2][:,1], image[mask2][:,2], c=image[mask2])
        # plt.show()
        savetxt(out_dir + outname, image[mask1])
