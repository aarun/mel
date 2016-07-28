import numpy as np
import os
from skimage import data
from skimage.io import imread
from PIL import Image
from skimage.segmentation import slic
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

for fn in file_list:
    if fn.endswith('jpg') :

		print 'Processing file: ', fn

		error = Image.open(fn.replace('.jpg', '_error.jpg'))
		original = Image.open(fn)
		imarr_err = np.array(error)
		imarr_orig = np.array(original)

		for i in range(len(imarr_err)):
			for j in range(len(imarr_err[i])):
				if (imarr_err[i][j][0] < 100 and imarr_err[i][j][1] < 100 and imarr_err[i][j][2] < 100) :
					imarr_orig[i][j] = [0,0,0]
				elif (imarr_err[i][j][0] > 190 and imarr_err[i][j][1] < 100 and imarr_err[i][j][2] < 100) :
					imarr_orig[i][j] = [0,0,0]

		orig_predict = Image.fromarray(imarr_orig)
		orig_predict.show()

		segments = slic(orig_predict, n_segments = 4, sigma = 1, slic_zero = 2)

		sp_dict = {}

		for (i, segVal) in enumerate(np.unique(segments)) :
			print i
			mask = np.zeros(segments.shape[:2], dtype='uint8')
			mask[segments == segVal] = 255
			sp_locations = mask[:,:] == 255
			area = len(mask[segments == segVal])

			r = (sum(imarr_orig[sp_locations,0]))/area
			g = (sum(imarr_orig[sp_locations,1]))/area
			b = (sum(imarr_orig[sp_locations,2]))/area

			sp_dict[segVal] = r + g + b

#		minimum = sp_dict[0]
#		min_ind = 0
#
#		min_2 = sp_dict[1]
#		min_2_ind = 1
#
#		for k in sp_dict :
#			if (sp_dict[k] < minimum) :
#				min_2 = minimum
#				min2_2_ind = min_ind
#				minimum = sp_dict[k]
#				min_ind = k
#			elif (sp_dict[k] < min_2) :
#				min_2 = sp_dict[k]
#				min_2_ind = k

		init = 0

		mask = np.zeros(segments.shape[:2], dtype = 'uint8')
		for k in sp_dict :
			mask[segments == k] = init
			init += 75

		post_image = Image.fromarray(mask)
		post_image.show()
