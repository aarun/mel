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
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')
args = vars(ap.parse_args())

file_list =[]



#forest = joblib.load('forest.pkl')
if (_platform == "darwin") : 
	with open('/Users/18AkhilA/Documents/mel/forest.pkl', 'rb') as f:
	    forest = cPickle.load(f)
else :
	with open('c:\mel\\forest.pkl', 'rb') as f:
	    forest = cPickle.load(f)





print args


if (args['list'] != None) :
	with open(args['list']) as batch_file :
		for line in batch_file :
			if line.endswith('.jpg\n') :
				line = line.replace('.jpg', '.txt')			
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)



for fn in file_list :
	if (fn.endswith('.txt')) :
		

		input_file = csv.DictReader(open(fn))
		print "Predicting " + fn
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

		temp = zip(r, g, b, dis, corr, con, en, hom, dtc)

		#data.extend(temp)




		prediction = forest.predict(temp)
		fn2 = fn.replace('.txt', '_Prediction.csv')
		#f_out=open(fn2,'w')
		#pred_string = ()

		count = 0

		w = csv.writer(open(fn2, "w"))
		for i in range(len(prediction)):

			w.writerow([prediction[i]])






