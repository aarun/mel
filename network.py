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
			a = a.strip('\r')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)


print file_list






tdata = []
tgroundtruth = []



data = []
gtc = 0
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


			temp = zip(r,g,b, dis, corr, con, en, hom, dtc, nrow, ncol)

			data.extend(temp)

			fn2 = fn.replace('.txt', '_Segmentation.txt')

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file = csv.DictReader(open(long_fn))

			#print ground_file.fieldnames

			keys = ground_file.fieldnames

			m = []

			for row in ground_file:
				m.append(int(row[" mask"]))
				tm = int(row[" mask"])
				if (tm == 1):
					gtc+=1


			groundtruth.extend(m)
			counter += 1

groundtruth = np.asarray(groundtruth)
data = np.asarray(data)

np.random.seed(1336)
shuffle = np.arange(len(groundtruth))
np.random.shuffle(shuffle)
data = data[shuffle]
groundtruth = groundtruth[shuffle]

scount = 0
for i in range(len(data)):
	if (groundtruth[i] == 1):
		tdata.append(data[i])
		tgroundtruth.append(groundtruth[i])
	elif (scount < gtc):
		tdata.append(data[i])
		tgroundtruth.append(groundtruth[i])
		scount+=1



model = Sequential()

model.add(Dense(11, init='uniform', input_dim = 11, activation='relu'))

model.add(Dense(6, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


tgroundtruth = np.asarray(tgroundtruth)
tdata = np.asarray(tdata)

np.random.seed(1337)
shuffle = np.arange(len(tgroundtruth))
np.random.shuffle(shuffle)
tdata = tdata[shuffle]
tgroundtruth = tgroundtruth[shuffle]


model.fit(tdata, tgroundtruth, nb_epoch=20, batch_size=10)

scores = model.evaluate(data, groundtruth)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model_json = model.to_json()
with open("networkrgb.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelrgb.h5")
print("Saved model to disk")








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










