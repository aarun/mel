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


ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required = False, help = 'number of files')
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')
args = vars(ap.parse_args())
number = None


if (args['number'] != None) :
	number = int(args['number'])

file_list =[]


if (args['list'] != None) :
	with open(args['list']) as batch_file :
		for line in batch_file :
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)


data = []
groundtruth = []

counter = 0
if (_platform == "darwin") : 
	gt_fn = '/Users/sahana/Mel/mel/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
else :
	gt_fn = 'C:\mel\ISBI2016_ISIC_Part3_Training_GroundTruth.csv'

		

for fn in file_list :
	if (fn.endswith('.jpg')) :
		if (number == None or counter < number) : 

			print 'loading ' + fn
			input_file = csv.DictReader(open(fn.replace('.jpg', '.txt')))

			features = []


			for row in input_file:
				r = []
				g = []
				b = []

				dis = []
				corr = []
				con = []
				en = []
				hom = []

				dtc = []
				nrow = []
				ncol = []


			for row in input_file:
				r.append(float(row[" Avg R value"]))
				g.append(float(row[" Avg G value"]))
				b.append(float(row[" Avg B value"]))

				dis.append(float(row[" Dissimilarity"]))
				corr.append(float(row[" Correlation"]))
				con.append(float(row[" Contrast"]))
				en.append(float(row[" Energy"]))
				hom.append(float(row[" Homogeneity"]))
				
				dtc.append(float(row[" Distance from center"]))
				nrow.append(float(row[" Normalized row"]))
				ncol.append(float(row[" Normalized column"]))

			temp = [r, g, b, dis, corr, con, en, hom, dtc, nrow, ncol]

			data.append(temp)
			counter += 1

m = []

with open(gt_fn) as gt :
	ground_file = csv.DictReader(gt)
	for row in ground_file :
		if (row["label"] == 'benign') :
			m.append(int(0))
		else :
			m.append(int(1))

groundtruth.extend(m)
print 'Label length: ' + str(len(groundtruth))
	
print 'TRAINING'
forest = RandomForestRegressor(n_estimators = 500, n_jobs = 8)
forest.fit(data, groundtruth)


if (_platform == "darwin") : 
	with open('/Users/sahana/mel/forest_part3.pkl', 'wb') as f:
	    cPickle.dump(forest, f)
else :
	with open('c:\mel\\forest_part3.pkl', 'wb') as f:
	    cPickle.dump(forest, f)