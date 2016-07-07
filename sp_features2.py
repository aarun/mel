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
from collections import defaultdict

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

for fn in os.listdir('.') :
    if fn.endswith('jpg') :
    	if counter < 10:
			#maskfn = 'ADD LOCATION'
			f_out=open(fn.replace('.png','.txt'),'w')

			image = Image.open(fn)
			assert image.mode == 'RGB'
			imarr_enc = np.array(image)
			imarr_dec = decodeSuperpixelIndex(imarr_enc)

			original = Image.open(fn)
			c2 = 0
			mask = None

			for fn2 in os.listdir('DIR OF MASKS') :
				if (counter == c2):
					mask = Image.open(fn2)
				c2 += 1

			imarr_mask = np.array(mask)
			imarr_orig = np.array(original)
			imarr_bw = rgb2gray(imarr_orig)

			segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)


			sp_dict = {}

			both = {} # distance in both directions


			xr = len(imarr_orig)
			yr = len(imarr_orig[0])

			#gt_dict = collections.Counter()
			#maskdict[i] = collections.Counter()

			#for i in range(0, xr):
			   # for j in range(0, yr): 
			    	#gt_dict += imarr_mask[i, j]

			    
			centerx = xr/2
			centery = yr/2

			for (i, segVal) in enumerate(np.unique(segments)) :
			#for (i, segVal) in enumerate([0,1]):	
				mask = np.zeros(segments.shape[:2], dtype='uint8')
				mask[segments == segVal] = 255
				area = len(mask[segments == segVal])	
				sp_locations = mask[:,:] == 255

				#if (gt_dict[segVal] / area > 127.5):
			    #   maskdict[segVal] = 1
			    #else:
			   	#	maskdict[segVal] = 0


				r = (sum(imarr_orig[sp_locations,0]))/area
				g = (sum(imarr_orig[sp_locations,1]))/area
				b = (sum(imarr_orig[sp_locations,2]))/area


				props = regionprops(mask, cache=True )

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

				distance = sqrt( abs(props[0].centroid[0] - (xr/2))**2 + abs(props[0].centroid[1] - (yr/2))**2 )


				sp_dict[segVal] = [props[0].centroid, area, r, g, b, dissimilarity, correlation, contrast, energy, homogeneity, distance]
			counter++
	

dict_str = ('Superpixel label, Centroid row, Centroid column, Area,'
	+ ' Avg R value, Avg G value, Avg B value, Dissimilarity, Correlation,'
	+ ' Contrast, Energy, Homogeneity, Distance from center' + '\n')

for k in sp_dict:
    dict_str += (str(k) + ', ' + str(int(sp_dict[k][0][0])) + ', ' 
    	+ str(int(sp_dict[k][0][1])) + ', ' + str(int(sp_dict[k][1]))
    	 + ', ' + str(sp_dict[k][2]) + ', ' + str(sp_dict[k][3])
    	  + ', ' + str(sp_dict[k][4])+ ', ' + str(sp_dict[k][5]) + ', ' 
    	  + str(sp_dict[k][6]) + ', ' + str(sp_dict[k][7]) + ', ' 
    	  + str(sp_dict[k][8]) + ', ' + str(sp_dict[k][9])  + ', ' + str(sp_dict[k][10]) + '\n')

f_out.write(dict_str)
f_out.close()

#imarr_diss = np.zeros((len(imarr_dec),len(imarr_dec[0])), dtype=np.uint8 )
#imarr_corr = np.zeros((len(imarr_dec),len(imarr_dec[0])), dtype=np.uint8 )
#imarr_cont = np.zeros((len(imarr_dec),len(imarr_dec[0])), dtype=np.uint8 )
#imarr_energy = np.zeros((len(imarr_dec),len(imarr_dec[0])), dtype=np.uint8 )
#imarr_homo =  np.zeros((len(imarr_dec),len(imarr_dec[0])), dtype=np.uint8 )

#for row in range(imarr_diss.shape[0]) :
#	for col in range(imarr_diss.shape[1]) :
#
#		imarr_diss[row][col] = int((sp_dict[segments[row][col]][5])*(25500/367))
#		if corr < 0:
#			corr = 0
#		imarr_corr[row][col] = corr
#		cont = int((sp_dict[segments[row][col]][7])*255)
#			cont = 255
#		imarr_cont[row][col] = cont
#		imarr_energy[row][col] = int((sp_dict[segments[row][col]][8])*255)
#		imarr_homo[row][col] = int((sp_dict[segments[row][col]][9])*255)


		
#img = Image.fromarray(imarr_diss)
#img2 = Image.fromarray(imarr_corr)
#img3 = Image.fromarray(imarr_cont)
#img4 = Image.fromarray(imarr_energy)
#img5 = Image.fromarray(imarr_homo)
#img.show()
#img2.show()
#img3.show()
#img4.show()
#img5.show()






