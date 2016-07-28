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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
import pylab as pl
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import argparse
from skimage.util import img_as_float
from skimage import io

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())

def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

		root_name = fn.strip('.jpg')
		prediction = np.loadtxt(root_name+'_Prediction.csv', delimiter= ",")

		for i in range(len(imarr_err)):
			for j in range(len(imarr_err[i])):
				if ((imarr_err[i][j][0] < 100) and (imarr_err[i][j][1] < 100) and imarr_err[i][j][2] < 100):
					imarr_orig[i][j][0] = 0
					imarr_orig[i][j][1] = 0
					imarr_orig[i][j][2] = 0


		#orig_predict = Image.fromarray(imarr_orig)
		imarr_bw = rgb2gray(imarr_orig)
		#orig_predict.show()
		#orig_bw = Image.fromarray(imarr_bw)
		#orig_bw.show()

		data = []

		segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)


		counter = 0
		input_file = csv.DictReader(open(fn.replace('.jpg', '.txt')))
		for row in input_file:
			if prediction[counter] < 0.5 :
				r = 0
				g = 0
				b = 0
			else :
				r = float(row[" Avg R value"])
				g = float(row[" Avg G value"])
				b = float(row[" Avg B value"])
			data.append([r, g, b])
			counter += 1

		l = len(data)
		#data = resize(l, 3)

		kmeans = KMeans(n_clusters = 3)
		kmeans.fit(data)

		labels = kmeans.labels_
		#labels.resize(l,1)

		
		color = np.array([0,0,0])
		count = np.array([0,0,0])

		imarr_predict = np.zeros((imarr_err.shape[0], imarr_err.shape[1]))


		for (i, segVal) in enumerate(np.unique(segments)) :
			currLabel = labels[segVal] 
			print str(currLabel) + '\n'
			imarr_predict[segments == segVal] = currLabel
			area = len(imarr_orig[segments == segVal])
			count[currLabel] += area


		color = color/count

		minimum = 0
		min_ind = 0
		min_2 = 1
		min_ind_2 = 1

		if color[1] < color[0] :
			minimum = color[1]
			min_ind = 1
			min_2 = color[0]
			min_ind_2 = 0
		if color[2] < minimum :
			min_2 = minimum
			min_ind_2 = min_ind
			minimum = color[2]
			min_ind = 2
		elif color[2] < min_2:
			min_2 = color[2]
			min_ind_2 = 2

		for i in range(len(imarr_err)):
			for j in range(len(imarr_err[i])):
				if imarr_predict[i][j] != min_ind_2:
					imarr_predict[i][j] = 0
				else :
					imarr_predict[i][j] = 255

		img = Image.fromarray(imarr_predict)
		img.show()
