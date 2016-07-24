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
	with open('/Users/18AkhilA/Documents/mel/forest_p1_ch.pkl', 'rb') as f:
	    forest = cPickle.load(f)
else :
	with open('c:\mel\\forest_p1_ch.pkl', 'rb') as f:
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

		r0 = []
		r1 = []
		r2 = []
		r3 = []
		r4 = []
		r5 = []
		r6 = []
		r7 = []

		g0 = []
		g1 = []
		g2 = []
		g3 = []
		g4 = []
		g5 = []
		g6 = []
		g7 = []

		b0 = []
		b1 = []
		b2 = []
		b3 = []
		b4 = []
		b5 = []
		b6 = []
		b7 = []

		dis = []
		corr = []
		con = []
		en = []
		hom = []
		
		dtc = []
		nrow = []
		ncol = []


		for row in input_file:
			r0.append(float(row[" R0"]))
			r1.append(float(row[" R1"]))
			r2.append(float(row[" R2"]))
			r3.append(float(row[" R3"]))
			r4.append(float(row[" R4"]))
			r5.append(float(row[" R5"]))
			r6.append(float(row[" R6"]))
			r7.append(float(row[" R7"]))

			g0.append(float(row[" G0"]))
			g1.append(float(row[" G1"]))
			g2.append(float(row[" G2"]))
			g3.append(float(row[" G3"]))
			g4.append(float(row[" G4"]))
			g5.append(float(row[" G5"]))
			g6.append(float(row[" G6"]))
			g7.append(float(row[" G7"]))

			b0.append(float(row[" B0"]))
			b1.append(float(row[" B1"]))
			b2.append(float(row[" B2"]))
			b3.append(float(row[" B3"]))
			b4.append(float(row[" B4"]))
			b5.append(float(row[" B5"]))
			b6.append(float(row[" B6"]))
			b7.append(float(row[" B7"]))

			dis.append(float(row[" Dissimilarity"]))
			corr.append(float(row[" Correlation"]))
			con.append(float(row[" Contrast"]))
			en.append(float(row[" Energy"]))
			hom.append(float(row[" Homogeneity"]))

			dtc.append(float(row[" Distance_center"]))
			nrow.append(float(row[" Norm_row"]))
			ncol.append(float(row[" Norm_column"]))

		temp = zip(r0, r1, r2, r3, r4, r5, r6, r7, g0, g1, g2, g3, g4, g5, g6, g7, b0, b1, b2, b3, b4, b5, b6, b7, dis, corr, con, en, hom, dtc, nrow, ncol)


		#data.extend(temp)




		prediction = forest.predict(temp)
		fn2 = fn.replace('.txt', '_Prediction.csv')
		#f_out=open(fn2,'w')
		#pred_string = ()


		w = csv.writer(open(fn2, "w"))
		for i in range(len(prediction)):

			w.writerow([prediction[i]])






