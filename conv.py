from keras.models import Sequential
from keras.layers import Dense, Activation, MaxoutDense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, Callback, History
from keras.optimizers import SGD
from keras.optimizers import RMSprop
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
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required = False, help = 'number of files')
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')

args = vars(ap.parse_args())
#args2 = vars(ap.parse_args())
number = None
if (args['number'] != None) :
	number = int(args['number'])

# notes which files to process
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













data = []
neighbors = []
groundtruth = []
counter = 0
counts = []
count1 = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/18AkhilA/Downloads/ISBI_2016_Part1_Training_GroundTruth-1'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'

		
for fn in file_list :
	if (fn.endswith('.txt')) :
		if (number == None or counter < number) : 

			# This section creates the feature vector to feed to the network. It reads the RGB value of
			# each superpixel and adds it to a list. 


			#print 'loading ' + fn

			input_file = csv.DictReader(open(fn))
			#print input_file.fieldnames

			r = []
			g = []
			b = []
			gray = []
			dis = []
			corr = []
			con = []
			en = []
			hom = []
			dtc = []




			for row in input_file:

				r.append(float(row[" Avg R value"]))
				red = float(row[" Avg R value"])
				#print r
				g.append(float(row[" Avg G value"]))
				green = float(row[" Avg G value"])
				b.append(float(row[" Avg B value"]))
				blue = float(row[" Avg B value"])
				dtc.append(float(row[" Distance from center"]))



				tempgray = 0.2989*red + 0.5870*green + 0.1140*blue #gray scale converter(not using it right now)

				gray.append(tempgray)




			temp = zip(r, g, b)

			data.extend(temp)

			fn2 = fn.replace('.txt', '_Segmentation.txt')

			fn3 = fn.replace('.txt', '_neighbors.txt')

			neigh_file = csv.DictReader(open(fn3))

			# This section reads a file which states the neighbors of each superpixel

			south = []
			north = []
			east = []
			west = []
			southeast = []
			northwest = []
			southwest = []
			northeast = []
			a = []

			tempn = []

			for row in neigh_file:
				a.append(int(row["label"]))
				south.append(int(row[" south"]))
				north.append(int(row[" north"]))
				east.append(int(row[" east"]))
				west.append(int(row[" west"]))
				southeast.append(int(row[" southeast"]))
				northwest.append(int(row[" northwest"]))
				northeast.append(int(row[" northeast"]))
				southwest.append(int(row[" southwest"]))

			tempn = zip(south, north, east, west, southeast, northwest, northeast, southwest, a)
			
			neighbors.extend(tempn)


			

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file = csv.DictReader(open(long_fn))

			# This section reads the groundtruth file

			keys = ground_file.fieldnames

			m = [keys[2]]
			

			for row in ground_file:
				m.append(int(row[" mask0"]))
				htemp = int(row[" mask0"])
				if (htemp == 1):
					count1 += 1

			counts.append(count1)



			groundtruth.extend(m)
			counter += 1
print "finish load. start placement", len(data), len(groundtruth), len(neighbors), len(counts)

#This part of the code creates a 3x3 grid of the RGB values of each superpixel and its neighbors
groundtruth = np.asarray(groundtruth)
data = np.asarray(data)

np.random.seed(1337)
shuffle = np.arange(len(groundtruth))
np.random.shuffle(shuffle)
data = data[shuffle]
#counts = counts[shuffle]
groundtruth = groundtruth[shuffle]



fulldata = []
fullground = []
checkcount = 0
count1 = len(data) #(2 * count1)/4
for i in range(len(data)):
	tempd = np.zeros((3, 3))

	factor = i - neighbors[i][8]

	tempd[2][1] = neighbors[i][0]+ factor
	tempd[0][1] = neighbors[i][1]+ factor
	tempd[1][2] = neighbors[i][2]+ factor
	tempd[1][0] = neighbors[i][3]+ factor
	tempd[2][2] = neighbors[i][4]+ factor
	tempd[0][0] = neighbors[i][5]+ factor
	tempd[0][2] = neighbors[i][6]+ factor
	tempd[2][0] = neighbors[i][7]+ factor
	tempd[1][1] = i

	

	tempf = np.zeros(( 3, 3, 3))

	for j in range(len(tempd)):
		for z in range(len(tempd[0])):
			index = int(tempd[j][z])
			if (index == -1) :
				tempf[j][z] = [0, 0, 0]
			else :
				tempf[j][z] = data[index]
	
	# This is the limiting thing I mentioned in my email. I am checking to see if the number of non-bordering
	# superpixels equal the number of bordering ones.
	if (groundtruth[i] == 1) :
		fulldata.append(tempf)
		fullground.append(groundtruth[i])
	else :
		if (checkcount < count1) :
			fulldata.append(tempf)
			fullground.append(groundtruth[i])
			checkcount +=1

fullground = np.asarray(fullground)
fulldata = np.asarray(fulldata)

np.random.seed(1338)
shuffle = np.arange(len(fullground))
np.random.shuffle(shuffle)
fulldata = fulldata[shuffle]
fullground = fullground[shuffle]

history = History()



# This part is the actual conv. net

print "starting fit"
model = Sequential()




model.add(ZeroPadding2D((1,1), input_shape=( 3, 3, 3))) #, input_shape=( 3, 3, 3)
model.add(Convolution2D(18, 3, 3, name='conv1_1'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#model.add(MaxPooling2D((2,2), strides=(2,2)))


#model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(18, 3, 3, name='conv2_1', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten(name="flatten"))
#model.add(Activation('sigmoid'))

print model.summary()

#model.add(Flatten(name="flatten"))
#model.add(Dense(64, init='uniform'))
#model.add(Activation('relu'))

model.add(Dense(1, activation='sigmoid'))



rms = RMSprop(lr=0.00001)

#binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

fulldata = np.asarray(fulldata)
fulldata = fulldata.reshape(fulldata.shape[0], 3, 3, 3)
print "starting fit"

model.fit(np.asarray(fulldata), np.asarray(fullground), nb_epoch=25, batch_size=10, validation_split=0.2)
scores = model.evaluate(np.asarray(fulldata), np.asarray(fullground))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# This part saves the network
model.save_weights("convnetworkfix.h5")
model_json = model.to_json()
with open("convnetworkfix.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
#model.save_weights("convnetworkdtc.h5")
print("Saved model to disk")





