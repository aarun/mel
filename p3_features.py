#Creates a CSV file with features for each superpixel
#Currently contains: superpixel label, area, centroid location, RGB, and 5 GLCM properties
#(GLCM properties: dissimilarity, correlation, contrast, energy, homogeneity)

import numpy as np
import os
from skimage.measure import regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.io import imread
from PIL import Image
from skimage.segmentation import slic
from math import sqrt
import collections
from collections import defaultdict
import argparse
from sys import platform as _platform

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())


file_list =[]

with open(args['list']) as batch_file :
	for line in batch_file :
		a = line.strip('\n')
		file_list.append(a)

print file_list

PATCH_SIZE = 10

def decodeSuperpixelIndex(rgbValue):
    """
    Decode an RGB representation of a superpixel label into its native scalar value.
    :param pixelValue: A single pixel, or a 3-channel image.
    :type pixelValue: numpy.ndarray of uint8, with a shape [3] or [n, m, 3]
    """
    return \
        (rgbValue[..., 0].astype(np.uint64)) + \
        (rgbValue[..., 1].astype(np.uint64) << np.uint64(8)) + \
        (rgbValue[..., 2].astype(np.uint64) << np.uint64(16))

# This may be used as:
from PIL import Image

def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

counter = 0

for fn in file_list:
    if fn.endswith('jpg') :
		f_out=open(fn.replace('.jpg','.txt'),'w')
		print 'Processing file: ', fn

		original = Image.open(fn)
		imarr_orig = np.array(original)
		imarr_bw = rgb2gray(imarr_orig)

		segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)

		sp_dict = {}

		img_row = len(imarr_orig)
		img_col = len(imarr_orig[0])
		half_diag = (sqrt((img_row**2) + (img_col**2)))/2

		for (i, segVal) in enumerate(np.unique(segments)) :

			mask2 = np.zeros(segments.shape[:2], dtype='uint8')
			mask2[segments == segVal] = 255
			area = len(mask2[segments == segVal])	
			sp_locations = mask2[:,:] == 255


			r = (sum(imarr_orig[sp_locations,0]))/area
			g = (sum(imarr_orig[sp_locations,1]))/area
			b = (sum(imarr_orig[sp_locations,2]))/area


			props = regionprops(mask2, cache=True )

			centroid_loc = [0,0]
			centroid_loc[0] = (int)(props[0].centroid[0])
			centroid_loc[1] = (int)(props[0].centroid[1])
			patch = imarr_bw[centroid_loc[0]:centroid_loc[0] + PATCH_SIZE, centroid_loc[1]:centroid_loc[1] + PATCH_SIZE]

			glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
			dissimilarity = greycoprops(glcm, 'dissimilarity')[0,0]
			correlation =  greycoprops(glcm, 'correlation')[0,0]
			contrast = greycoprops(glcm, 'contrast')[0,0]
			energy = greycoprops(glcm, 'energy')[0,0]
			homogeneity = greycoprops(glcm, 'homogeneity')[0,0]

			distance = (sqrt( abs(props[0].centroid[0] - (img_row/2))**2 + abs(props[0].centroid[1] - (img_col/2))**2 ))/half_diag
			nrow = float(props[0].centroid[0]/img_row)
			ncol = float(props[0].centroid[0]/img_col)

			sp_dict[segVal] = [props[0].centroid, area, r, g, b, dissimilarity, correlation, contrast, energy, homogeneity, distance, nrow, ncol]

			dict_str = ('Superpixel label, Centroid row, Centroid column, Area,'
				+ ' Avg R value, Avg G value, Avg B value, Dissimilarity, Correlation,'
				+ ' Contrast, Energy, Homogeneity, Distance from center,'
				+ ' Normalized row, Normalized column' + '\n')


		for k in sp_dict:
			dict_str += (str(k) + ', ' + str(int(sp_dict[k][0][0])) + ', ' 
				+ str(int(sp_dict[k][0][1])) + ', ' + str(int(sp_dict[k][1]))
				 + ', ' + str(sp_dict[k][2]) + ', ' + str(sp_dict[k][3])
				  + ', ' + str(sp_dict[k][4])+ ', ' + str(sp_dict[k][5]) + ', ' 
				  + str(sp_dict[k][6]) + ', ' + str(sp_dict[k][7]) + ', ' 
				  + str(sp_dict[k][8]) + ', ' + str(sp_dict[k][9])  + ', '
				   + str(sp_dict[k][10]) + ', ' + str(sp_dict[k][11]) + ', '
				   	+ str(sp_dict[k][12]) + '\n')

		f_out.write(dict_str)
		f_out.close()

