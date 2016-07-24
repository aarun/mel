from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
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
from keras.models import model_from_json
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required = False, help = 'number of files')
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')

args = vars(ap.parse_args())
#args2 = vars(ap.parse_args())
number = None
if (args['number'] != None) :
	number = int(args['number'])


file_list =[]

checker = []

if (args['list'] != None) :
	with open(args['list']) as batch_file :
		for line in batch_file :
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)

filename = '/Users/18AkhilA/Downloads/ISBI2016_ISIC_Part1_Training_ch/network.json'
filename2 = '/Users/18AkhilA/Downloads/ISBI2016_ISIC_Part1_Training_ch/model.h5'
json_file = open(filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(filename2)
print("Loaded model from disk")










data = []
groundtruth = []
counter = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/18AkhilA/Downloads/ISBI_2016_Part1_Training_GroundTruth'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'

		
for fn in file_list :
	if (fn.endswith('.txt')) :
		if (number == None or counter < number) : 


			print 'loading ' + fn

			input_file = csv.DictReader(open(fn))
			#print input_file.fieldnames

			r0 = []
			r1 = []
			r2 = []
			r3 = []
			r4 = []
			r5 = []
			r6 = []
			r7 = []

			data = []

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


			temp = zip(r0,r1,r2,r3,r4,r5,r6,r7, g0,g1,g2,g3,g4,g5,g6,g7, b0,b1,b2,b3,b4,b5,b6,b7, dis, corr, con, en, hom, dtc, nrow, ncol)

			data.extend(temp)

			counter += 1

			prediction = model.predict(data)

			fn2 = fn.replace('.txt', '_Prediction.csv')


			w = csv.writer(open(fn2, "w"))
			for i in range(len(prediction)):

				w.writerow([prediction[i][0]])








#prediction = model.predict(data2)



#f_out=open(fn2,'w')
#pred_string = ()

#count = 0

#w = csv.writer(open(fn3, "w"))
#for i in range(len(prediction)):

	
#	w.writerow([prediction[i][0]])


	



#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#model.add(Convolution3D(nb_filter= 0, input_dim = (11, X, Y)))

#model.add(MaxPooling3D(pool_size=(1, 1, numfeatures)))










