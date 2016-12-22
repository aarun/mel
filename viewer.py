from sklearn.ensemble import RandomForestRegressor
import numpy as np
from PIL import Image
import argparse
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import csv
import cPickle
from sklearn.externals import joblib
import os

from sys import platform as _platform


if (_platform == "darwin") : 
		seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_Data'
else :
		seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_Data'

for fn in os.listdir('.') :
	if (fn.endswith('.csv')) :
		
		input_file = np.loadtxt(fn, delimiter= ",")
		fn2 = fn.replace('_Prediction.csv', '.jpg')
		print "Displaying Prediction of " + fn2
		long_fn = seg_gt_dir + "/" + fn2

		image = Image.open(long_fn)
		imarr_image = np.array(image)
		imarr_mask = np.zeros((imarr_image.shape[0], imarr_image.shape[1]))
		segments = slic(image, n_segments = 3000, sigma = 5, slic_zero = 2)

		for (i, segVal) in enumerate(np.unique(segments)) :

			val = input_file[segVal]

			if (val < 0.5) :
				imarr_mask[segments == segVal] = 0
			else :
				imarr_mask[segments == segVal] = 255


		im = Image.fromarray(imarr_mask)
		im.show()



			
