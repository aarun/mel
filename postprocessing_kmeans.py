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

#print file_list
print 'image , accuracy, sensitivity, specificity, dice_coeff, jaccard_ind'
for fn in file_list:
    if fn.endswith('jpg'):

		#print 'Processing file: ', fn

		error = Image.open(fn.replace('.jpg', '_error.jpg'))
		original = Image.open(fn)
		imarr_err = np.array(error)
		imarr_orig = np.array(original)
		imarr_overlay = np.array(original)

		root_name = fn.strip('.jpg')
		prediction = np.loadtxt(root_name+'_Prediction.csv', delimiter= ",")

		if (_platform == "darwin") : 
			seg_gt_dir = '/users/sahana/mel/ISBI2016_ISIC_Part1_Test_GroundTruth'
			orig_dir = '/Users/sahana/Mel/ISBI2016_ISIC_Part1_Test_Data/'
			gt_fn = seg_gt_dir + "/" + root_name + '_Segmentation.png'			
		else :
			seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Test_GroundTruth'
			orig_dir = 'C:\mel\ISBI2016_ISIC_Part1_Test_Data\\'			
			gt_fn = seg_gt_dir + "\\" + root_name + '_Segmentation.png'	

		gt = Image.open(gt_fn,'r')
		imarr_gt = np.array(gt)

		for i in range(len(imarr_orig)):
			for j in range(len(imarr_orig[i])):
				if ((imarr_err[i][j][0] < 100) and (imarr_err[i][j][1] < 100) and imarr_err[i][j][2] < 100):
					imarr_orig[i][j][0] = 0
					imarr_orig[i][j][1] = 0
					imarr_orig[i][j][2] = 0
				elif ((imarr_err[i][j][0] > 200) and (imarr_err[i][j][1] < 100) and imarr_err[i][j][2] < 100):
					imarr_orig[i][j][0] = 0
					imarr_orig[i][j][1] = 0
					imarr_orig[i][j][2] = 0


		#orig_predict = Image.fromarray(imarr_orig)
		imarr_bw = rgb2gray(imarr_orig)
		#orig_predict.show()
		orig_bw = Image.fromarray(imarr_bw)
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

		imarr_predict = np.zeros((imarr_orig.shape[0], imarr_orig.shape[1]))



		for (i, segVal) in enumerate(np.unique(segments)) :
			currLabel = labels[segVal] 
			imarr_predict[segments == segVal] = currLabel
			area = len(imarr_orig[segments == segVal])
			count[currLabel] += area

		sp_locations = imarr_predict[:,:] == 0
		color[0] = sum(imarr_bw[sp_locations])
		sp_locations = imarr_predict[:,:] == 1
		color[1] = sum(imarr_bw[sp_locations])
		sp_locations = imarr_predict[:,:] == 2
		color[2] = sum(imarr_bw[sp_locations])

		color = color/count

		minimum = min(color)
		min_ind = -1
		min_2 = -1
		min_ind_2 = -1

		if color[0] == minimum:
			min_ind = 0
		elif color[1] == minimum:
			min_ind = 1
		else:
			min_ind = 2

		if min_ind == 0 and color[1] < color[2]:
			min_2 = color[1]
			min_ind_2 = 1
		elif min_ind == 0:
			min_2 = color[2]
			min_ind_2 = 2
		if min_ind == 1 and color[0] < color[2]:
			min_2 = color[0]
			min_ind_2 = 0
		elif min_ind == 1:
			min_2 = color[2]
			min_ind_2 = 2
		if min_ind == 2 and color[0] < color[1]:
			min_2 = color[0]
			min_ind_2 = 0
		elif min_ind == 2:
			min_2 = color[1]
			min_ind_2 = 1

		combine = 0
		if color[3-min_ind-min_ind_2] - color[min_ind_2] < 60 :
			combine = 1
		
		#print str(color)

		for i in range(len(imarr_err)):
			for j in range(len(imarr_err[i])):
				if imarr_predict[i][j] == min_ind:
					imarr_predict[i][j] = 0
				elif imarr_predict[i][j] == min_ind_2:
					imarr_predict[i][j] = 255
				elif combine == 1 and imarr_predict[i][j] == 3-min_ind-min_ind_2:
					imarr_predict[i][j] = 255
				elif imarr_predict[i][j] == 3-min_ind-min_ind_2:
					imarr_predict[i][j] = 0
				#x = int(imarr_predict[i][j])
				#a = color[x]
				#imarr_predict[i][j] = a

		#print str(color[3-min_ind-min_ind_2] - color[min_ind_2]) 

		falsepos = 0
		falseneg = 0
		trueneg = 0
		truepos = 0

		for i in range(len(imarr_gt)) :
			for j in range(len(imarr_gt[i])) :
				if (imarr_gt[i][j] == 255):
					if (imarr_predict[i][j] == 0) :
						falseneg += 1
						imarr_overlay[i][j] = [255,0,0]
					else :
						truepos += 1
						imarr_overlay[i][j] = [255,255,255]
				else :
					if (imarr_predict[i][j] == 0) :
						trueneg += 1
						imarr_overlay[i][j] = [0,0,0]
					else :
						falsepos += 1
						imarr_overlay[i][j] = [0,255,0]

		tp = float(truepos)
		tn = float(trueneg)
		fp = float(falsepos)
		fn = float(falseneg)

		accuracy = (tp + tn)/(tp + fp + tn + fn)
		sensitivity = tp/(tp + fn)
		specificity = tn/(tn + fp)
		dice_coeff = (2*tp)/((2*tp) + fn + tp)
		jaccard_ind = tp/(tp + fn + fp)

		image = Image.fromarray(imarr_overlay)
		image2 = image.convert('RGB')
		image2.save(root_name + '_error_post.png')	

		print root_name, ',', accuracy, ',', sensitivity, ',', specificity, ',', dice_coeff, ',', jaccard_ind


