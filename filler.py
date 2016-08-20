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
from scipy import ndimage
import matplotlib.pyplot as plt
import csv

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())


file_list =[]

with open(args['list']) as batch_file :
	for line in batch_file :
		if line.endswith('.jpg\n') :
			line = line.replace('.jpg', '.txt')
		a = line.strip('\n')
		if a.endswith('\r') :
			a = a.strip('\r')
		file_list.append(a)

print file_list

for fn in file_list:
    if fn.endswith('.txt'):


		root_name = fn.strip('.txt')
		#print 'Processing file root: ', root_name		
		prediction = np.loadtxt(root_name+'_Prediction_cnn2.csv', delimiter= ",")

		neighbors = []

		with open(root_name+'_neighbors.txt') as f:
			lines = (line for line in f if not line.startswith('#'))
			neighbors.append(np.loadtxt(lines, delimiter=',', skiprows=1))

		#neighbors = np.loadtxt(root_name+'_neighbors.txt', delimiter= ",")
		print fn

		check = 0
		for i in range(len(prediction)):
			if (prediction[i] < 0.5):
				prediction[i] = 0
			else:
				prediction[i] = 1

		for i in range(len(prediction)):
			check = 0
			val = prediction[i]
			if (val == prediction[neighbors[0][i][0]]):
				check+=1
			if (val == prediction[neighbors[0][i][1]]):
				check+=1
			if (val == prediction[neighbors[0][i][2]]):
				check+=1
			if (val == prediction[neighbors[0][i][3]]):
				check+=1
			if (val == prediction[neighbors[0][i][4]]):
				check+=1
			if (val == prediction[neighbors[0][i][5]]):
				check+=1
			if (val == prediction[neighbors[0][i][6]]):
				check+=1
			if (val == prediction[neighbors[0][i][7]]):
				check+=1

			if(check <= 3):
				if (val == 0):
					prediction[i] = 1
				else:
					prediction[i] = 0


		w = csv.writer(open(root_name+'_Prediction_cnn3.csv', "w"))
		for i in range(len(prediction)):

			w.writerow([prediction[i]])







    	






















"""
    	original = Image.open(fn)

    	imarr_org = np.array(original)
    	new_arr = np.zeros((imarr_org.shape[0], imarr_org.shape[1]))

    	for i in range(len(imarr_org)):
    		for j in range(len(imarr_org[0])):
    			if (imarr_org[i][j][0] == 255 and imarr_org[i][j][1] == 0):
    				imarr_org[i][j] = [0, 0, 0]
    				new_arr[i][j] = 0
    			elif (imarr_org[i][j][0] == 0 and imarr_org[i][j][1] == 255):
    				imarr_org[i][j] = [255, 255, 255]
    				new_arr[i][j] = 1
    			else :
    				new_arr[i][j] = imarr_org[i][j][0]

    	imarr_new = ndimage.binary_fill_holes(new_arr).astype(int)

    	imarr_new = np.array(imarr_new * 255).astype('uint8')

    	


    	image = Image.fromarray(imarr_new)

    	image.save(fn)

"""


