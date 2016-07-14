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
import json

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required = False, help = 'number of files')
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')
args = vars(ap.parse_args())
#args2 = vars(ap.parse_args())
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


	#print args

data = []
groundtruth = []
groundtruth2 = []
counter = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part2_Training_GroundTruth'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part2_Training_GroundTruth'

		
for fn in file_list :
	if (fn.endswith('.txt')) :
		if (number == None or counter < number) : 


			print 'loading ' + fn

			input_file = csv.DictReader(open(fn))
			#print input_file.fieldnames

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


			temp = zip(r, g, b, dis, corr, con, en, hom, dtc, nrow, ncol)

			data.extend(temp)

			fn2 = fn.replace('.txt', '.json')

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file {}

			with open(long_fn) as gt :
				ground_file = json.load(gt)

			#print ground_file.fieldnames

			m = [0]
			n = [0]

			for i in range(len(ground_file['globules'])) :
				m.append(ground_file['globules'][i])
				n.append(ground_file['streaks'][i])

			groundtruth.extend(m)
			groundtruth2.extend(n)
			counter += 1



	
			
			

print 'TRAINING'
forest = RandomForestRegressor(n_estimators = 500, n_jobs = 8)
forest.fit(data, groundtruth)

forest2 = RandomForestRegressor(n_estimators = 500, n_jobs = 8)
forest2.fit(data, groundtruth2)



if (_platform == "darwin") : 
	with open('/Users/sahana/mel/forest_part2_globules.pkl', 'wb') as f:
	    cPickle.dump(forest, f)
else :
	with open('c:\mel\\forest_part2_globules.pkl', 'wb') as f:
	    cPickle.dump(forest, f)

if (_platform == "darwin") : 
	with open('/Users/sahana/mel/forest_part2_streaks.pkl', 'wb') as f:
	    cPickle.dump(forest2, f)
else :
	with open('c:\mel\\forest_part2_streaks.pkl', 'wb') as f:
	    cPickle.dump(forest2, f)














