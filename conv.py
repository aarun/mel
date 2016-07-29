from keras.models import Sequential
from keras.layers import Dense, Activation, MaxoutDense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
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
from keras.utils import np_utils

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
			if('\r' in a) :
				a = a.strip('\r')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)







print len(file_list)





data = []
neighbors = []
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
			#print len(tempn)
			neighbors.extend(tempn)


			

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file = csv.DictReader(open(long_fn))

			#print ground_file.fieldnames

			keys = ground_file.fieldnames

			m = [keys[2]]

			for row in ground_file:
				m.append(int(row[" mask0"]))

			groundtruth.extend(m)
			counter += 1
print "finish load. start placement", len(data), len(groundtruth), len(neighbors)
fulldata = []

for i in range(len(data)):
	tempd = np.zeros((3, 3))
	#print i

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





#print groundtruth



print "starting fit"
model = Sequential()

model.add(UpSampling2D(size=(3, 3), input_shape=( 3, 3, 3)))
model.add(ZeroPadding2D((1,1))) 
model.add(Convolution2D(16, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(16, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(8, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(8, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3_1'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3_2'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3_3'))
#model.add(MaxPooling2D((2,2), strides=(2,2)))

#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(128, 3, 3, activation='relu', name='conv4_1'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(128, 3, 3, activation='relu', name='conv4_2'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(128, 3, 3, activation='relu', name='conv4_3'))
#model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten(name="flatten"))
model.add(Dense(10, activation='relu', name='dense_1'))
model.add(Dropout(0.5))
#model.add(Dense(102, activation='relu', name='dense_2'))
#model.add(Dropout(0.5))
model.add(Dense(1, name='dense_3'))
model.add(Activation("sigmoid",name="sigmoid"))

print model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "starting fit"

model.fit(np.asarray(fulldata), np.asarray(groundtruth), nb_epoch=2, batch_size=10)

scores = model.evaluate(np.asarray(fulldata), np.asarray(groundtruth))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


"""

model.add(UpSampling2D(size=(2, 2), input_shape=( 3, 3, 3)))

#model.add(ZeroPadding2D(padding=(0, 1, 1)))


model.add(Convolution2D(nb_filter=6, nb_row = 3, nb_col = 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))


#model.add(UpSampling2D(size=(2, 2)))

model.add(Convolution2D(nb_filter=3, nb_row = 3, nb_col = 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

model.add(Convolution2D(1, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Flatten())

#model.add(Dense(30))#, activation='softmax'))
#model.add(Activation('relu'))

model.add(Dense(3))#, activation='softmax'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

model.add(Dense(1))#, activation='softmax'))
model.add(Activation('sigmoid'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

#model.add(Activation('softmax'))
print model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "starting fit"

model.fit(np.asarray(fulldata), np.asarray(groundtruth), nb_epoch=25, batch_size=10)

scores = model.evaluate(np.asarray(fulldata), np.asarray(groundtruth))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

"""

model_json = model.to_json()
with open("convnetwork.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("convnetwork.h5")
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










