from keras.models import Sequential
from keras.layers import Dense, Activation, MaxoutDense
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import ActivityRegularization
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.regularizers import l2, activity_l2
import theano
from theano import tensor as T
from keras.models import Model
#from keras.models import save_model
from keras.layers import Merge
import numpy as np
import pickle
import argparse
from skimage import io
import csv
from sklearn.externals import joblib
import os
from sys import platform as _platform
from keras.models import model_from_json
from PIL import Image
import h5py
from keras.utils import np_utils
from keras import backend as K

o



json_file = open(filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(filename2)

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required = False, help = 'number of files')
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')

args = vars(ap.parse_args())
#args2 = vars(ap.parse_args())
number = None
if (args['number'] != None) :
	number = int(args['number'])

number = 2
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
counter = 0


for fn in file_list :
	if (fn.endswith('neighbors.txt')):
		print "wrong"

	elif (fn.endswith('.jpg') and not fn.endswith('error.jpg')) :
		if (number == None or counter < number) : 

			print 'loading ' + fn
			fn2 = fn.replace('.jpg', '_Prediction.jpeg')

			dat = []

			im = Image.open(fn)



			imarr = np.array(im)

			img = im.resize((1024, 768), Image.ANTIALIAS)

			#img.show()

			temp = np.array(img)
			print len(imarr), len(imarr[0]), len(temp), len(temp[0])

			dat.append(temp)

			dat = np.asarray(dat)

			dat = dat.reshape( dat.shape[0], 3, 768, 1024)

			prediction = model.predict(dat)

			prediction = np.asarray(prediction)

			pred = np.zeros((768, 1024))

			print len(prediction[0][0][0])

			for i in range(0, len(pred)) :
				for j in range(0, len(pred[0])) :
					#print prediction[i][j]
					f = prediction[0][0][i][j]
					if (f > 0.5):
						pred[i][j] = 255
					else :
						pred[i][j] = 0

			x = len(imarr)
			y = len(imarr[0])
			print x, y

			

			image = Image.fromarray(pred)
			#image = image.resize((y, x), Image.ANTIALIAS)
			image = image.convert('RGB')

			image.show()

			#image.save(fn2, 'JPEG')
			counter +=1











