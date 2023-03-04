#Refs used
#https://scikit-image.org/docs/

import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
from sklearn.mixture import GaussianMixture
from skimage import morphology
from skimage.measure import label, regionprops
# import skimage

img_dir = "ECE5242Proj1-test/"
out_dir = "img-out/"

table = np.load("lookup.npy")
def lookup(x):
    # return 0
    val = table[int(x[0] * 255 // 4)][int(x[1] * 255 // 4)][int(x[2] * 255 // 4)]
    return val

# Manual loop through pixels, achieves same output as np.apply_along_axis
# No longer used as lookup table is created instead, but left in for clarity
# for filename in os.listdir(img_dir):
#     if filename.endswith(".png"):
#         image = plt.imread(img_dir + "train_20_dist8.png")
#         # image = plt.imread(img_dir + filename)
#         mask = np.zeros([800, 600])
#         for row in range(len(image)):
#             for val in range(len(image[row])):
#                 temp = bayes(image[row][val])
#                 mask[row][val] = temp > 0.9

#         erode = morphology.binary_erosion(mask, morphology.disk(5))
#         dilate = morphology.binary_dilation(erode, morphology.disk(5))

#         fig = plt.figure()
#         plt.imshow(dilate)
#         label_img = label(dilate)
#         regions = regionprops(label_img)
#         for props in regions:
#             if(math.abs(props.bbox[2] - props.bbox[0]) < math.abs(props.bbox[3] - props.bbox[1])):
#                 continue
#             minr, minc, maxr, maxc = props.bbox
#             bx = (minc, maxc, maxc, minc, minc)
#             by = (minr, minr, maxr, maxr, minr)
#             plt.plot(bx, by, '-b', linewidth=2.5)
#         plt.show()

for filename in os.listdir(img_dir):
    if filename.endswith(".png"):
        # Took np functions from Vikram
        print("Analyzing " + filename + "...")
        image = plt.imread(img_dir + filename)
        mask = np.apply_along_axis(lookup, 2, image)
        mask = mask > 0.999999

        erode = morphology.binary_erosion(mask, morphology.disk(4))
        dilate = morphology.binary_dilation(erode, morphology.disk(8))
        fig = plt.figure()
        plt.imshow(image)

        label_img = label(dilate)
        regions = regionprops(label_img)
        for props in regions:
            minY, minX, maxY, maxX = props.bbox
            if(abs(maxY - minY) <= abs(maxX - minX)):
                continue
            bx = (minX, maxX, maxX, minX, minX)
            by = (minY, minY, maxY, maxY, minY)
            dist = 1074.1 * abs(maxY - minY) ** -1.026
            plt.plot(bx, by, '-b', linewidth=1.5)
            plt.text(props.bbox[1], props.bbox[0] - 30, 'dist: {:.2f}ft at x = {:.0f}, y = {:.0f}'.format(dist, props.centroid[0], props.centroid[1]), fontsize = 8, bbox = dict(facecolor = 'red', alpha = 0.5))
        # plt.imshow(mask)
        # plt.show()
        plt.savefig(out_dir + filename)
        plt.close()
        print("Done!")