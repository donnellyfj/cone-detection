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
from skimage.measure import label, regionprops, regionprops_table
from scipy.stats import norm
# import skimage

data_dir = "train-out/"
data_dir2 = "train-out2/"
red_dir = "train-outRed/"
img_dir = "ECE5242Proj1-trainRed/"
out_dir = "img-out/"
data = np.array([])
data2 = np.array([])
dataRed = np.array([])
i = 0
j = 0
print("Loading training data...")
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        if not os.path.exists(data_dir2 + filename):
            continue
        temp = np.loadtxt(data_dir + filename)
        if len(data) == 0:
            data = temp
        else:
            data = np.concatenate((data, temp))
        # print(data)

        temp = np.loadtxt(data_dir2 + filename)
        if len(data2) == 0:
            data2 = temp
        else:
            data2 = np.concatenate((data2, temp))
    i += 1
    # if i > 15:
    #     break
for filename in os.listdir(red_dir):
    if filename.endswith(".csv"):
        if not os.path.exists(red_dir + filename):
            continue
        temp = np.loadtxt(red_dir + filename)
        if len(dataRed) == 0:
            dataRed = temp
        else:
            dataRed = np.concatenate((dataRed, temp))
    j += 1

meanCone = np.mean(data, axis=0)
meanCone = np.atleast_2d(meanCone).T
print(meanCone[:,0])
varianceCone = np.cov(data.T)

meanNot = np.mean(data2, axis=0)
meanNot = np.atleast_2d(meanNot).T
varianceNot = np.cov(data2.T)

# meanRed = np.mean(dataRed, axis=0)
# meanRed = np.atleast_2d(meanRed).T
# varianceRed = np.cov(dataRed.T)

print(meanCone)
print("Vars")
print(data.shape)

print(varianceCone)
print("Done!")

# Functions to calculate values manually, produce the same values as np functions
# Calculate Mean Manually
# sum = np.zeros([3,1])
# print("Sum")
# print(sum)
# for val in data:
#     # print(np.atleast_2d(val - mean).T)
#     val = np.atleast_2d(val).T
#     sum += val
#     # break
# sum /= len(data)
# print(sum)
# mean = sum

# Calculate Variance Manually
# sum = np.zeros([3,3])
# print("Sum")
# print(sum)
# for val in data:
#     # print(np.atleast_2d(val - mean).T)
#     val = np.atleast_2d(val).T
#     sum += np.matmul(val - mean, (val - mean).T)
#     # break
# sum /= len(data)
# print(sum)
# variance = sum

def multiGauss(x, mean, variance):
    term1 = 1.0 / math.sqrt((2 * math.pi) * np.linalg.det(variance))
    invVar = np.linalg.inv(variance)
    x = np.atleast_2d(x).T
    prod1 = np.matmul(-(x - mean).T, np.atleast_2d(invVar))
    term2 = np.matmul(prod1, np.atleast_2d(x - mean))
    return term1 * math.exp(term2)

def bayes(x):
    prXY = multiGauss(x, meanCone, varianceCone)
    prXY2 = multiGauss(x, meanNot, varianceNot)
    # prXYRed = multiGauss(x, meanRed, varianceRed)

    prY1 = len(data) / (600 * 800 * i)
    prY2 = len(data2) / (600 * 800 * i)
    prYRed = len(dataRed) / (600 * 800 * j)

    prX = prXY * prY1 + prXY2 * prY2 #+ prXYRed * prYRed

    if prX == 0:
        prYX = 0
    else:
        prYX = prXY * prY1 / prX

    return prYX

print("Generating lookup table...")
lookup = np.zeros([64,64,64])
for r in range(64):
        for g in range(64):
            for b in range(64):
                # pixel = np.array([[r << 2], [g << 2], [b << 2]])
                lookup[r][g][b] = bayes([r * 4 / 255, g * 4 / 255, b * 4 / 255])

multiConeR = norm(meanCone[0,0], varianceCone[0, 0])
multiConeG = norm(meanCone[1,0], varianceCone[1, 1])
multiConeB = norm(meanCone[2,0], varianceCone[2, 2])
multiNotR = norm(meanNot[0,0], varianceNot[0, 0])
multiNotG = norm(meanNot[1,0], varianceNot[1, 1])
multiNotB = norm(meanNot[2,0], varianceNot[2, 2])

#Plot Gaussian
x = np.linspace(0, 1, 100)
plt.title('Cone PDFs')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.plot(x, multiConeR.pdf(x), '-r')
plt.plot(x, multiNotR.pdf(x), '--r')
plt.plot(x, multiConeG.pdf(x), '-g')
plt.plot(x, multiNotG.pdf(x), '--g')
plt.plot(x, multiConeB.pdf(x), '-b')
plt.plot(x, multiNotB.pdf(x), '--b')
plt.show()


np.save("lookup.npy", lookup)
print("Done!")