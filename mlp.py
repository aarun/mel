
import numpy as np
from PIL import Image
from skimage import io
from skimage.util import img_as_float
import csv
import cPickle
import os

import sklearn.neural_network
from sys import platform as _platform

from sklearn.neural_network import MLPClassifier

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


data = []
groundtruth = []
counter = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/sahana/Mel/mel/ISBI2016_ISIC_Part1_Training_GroundTruth_mlp'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth_mlp'

if (_platform == "darwin") : 
	train_dir = '/Users/sahana/Mel/mel/ISBI2016_ISIC_Part1_Training_Data_mlp'
else :
	train_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_Data_mlp'

		
for fn in file_list :
	if (fn.endswith('.txt')) :
		if (number == None or counter < number) : 

			if (_platform == "darwin") : 
				train_data_fn = train_dir + "/" + fn
			else :
				train_data_fn = train_dir + '\\' + fn2

			print 'loading ' + fn
			input_file = csv.DictReader(open(train_data_fn))

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

			fn2 = fn.replace('.txt', '_Segmentation.txt')

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file = csv.DictReader(open(long_fn))
			m = []

			for row in ground_file:
				m.append(int(row[" mask0"]))

			groundtruth.extend(m)
			counter += 1

print 'Starting training.'
mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
mlp.fit(data, groundtruth) 
MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(8, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)


if (_platform == "darwin") : 
	test_dir = '/Users/sahana/Mel/mel/ISBI2016_ISIC_Part1_Test_Data_mlp'
else :
	test_dir = 'C:\mel\ISBI2016_ISIC_Part1_Test_Data_mlp'

for fn in os.listdir('.') :
	if (fn.endswith('.txt')) :
		print 'Testing' + fn
		test_file = csv.DictReader(open(fn))

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

		for row in test_file :
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
		predction = mlp.predict(temp)
		fn2 = fn.replace('.txt', '_Prediction.csv')

		w = csv.writer(open(fn2, "w"))
		for i in range(len(prediction)):

			w.writerow([prediction[i]])


