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

filename = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_Data/convnetwork.json'
filename2 = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_Data/convnetwork.h5'
json_file = open(filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(filename2)
print("Loaded model from disk")










data = []
groundtruth = []
neighbors = []
counter = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/18AkhilA/Downloads/ISBI_2016_Part1_Test_GroundTruth'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'

		
for fn in file_list :
	if (fn.endswith('.txt')) :
		if (number == None or counter < number) : 


			print 'loading ' + fn

			input_file = csv.DictReader(open(fn))
			#print input_file.fieldnames

			r = []
			g = []
			b = []




			for row in input_file:
				r.append(float(row[" Avg R value"]))
				g.append(float(row[" Avg G value"]))
				b.append(float(row[" Avg B value"]))



			temp = zip(r,g,b)

			data.extend(temp)

			fn2 = fn.replace('.txt', '_Segmentation.txt')

			fn3 = fn.replace('.txt', '_neighbors.txt')

			neigh_file = csv.DictReader(open(fn3))

			south = []
			north = []
			east = []
			west = []
			southeast = []
			northwest = []
			southwest = []
			northeast = []

			tempn = []

			for row in neigh_file:
				south.append(int(row[" south"]))
				north.append(int(row[" north"]))
				east.append(int(row[" east"]))
				west.append(int(row[" west"]))
				southeast.append(int(row[" southeast"]))
				northwest.append(int(row[" northwest"]))
				northeast.append(int(row[" northeast"]))
				southwest.append(int(row[" southwest"]))

			tempn = zip(south, north, east, west, southeast, northwest, northeast, southwest)
			neighbors.extend(tempn)


			

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			counter += 1

			fulldata = []

			for i in range(len(data)):
				tempd = np.zeros((3, 3))

				tempd[2][1] = neighbors[i][0]
				tempd[0][1] = neighbors[i][1]
				tempd[1][2] = neighbors[i][2]
				tempd[1][0] = neighbors[i][3]
				tempd[2][2] = neighbors[i][4]
				tempd[0][0] = neighbors[i][5]
				tempd[0][2] = neighbors[i][6]
				tempd[2][0] = neighbors[i][1]
				tempd[1][1] = i

				#print tempd

				tempf = np.zeros(( 3, 3, 3))

				for j in range(len(tempd)):
					for z in range(len(tempd[0])):
						index = int(tempd[j][z])
						if (index == -1) :
							tempf[j][z] = [0, 0, 0]
						else :
							tempf[j][z] = data[index]

				fulldata.append(tempf)

			print "predicting"

			prediction = model.predict(np.asarray(fulldata))

			print "done predicting"

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










