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

class Deconvolution2D(Convolution2D):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, nb_row, nb_col)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_row, nb_col, nb_filter)` if dim_ordering='tf'.
    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegular            print(single_image.shape)izer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, binded_conv_layer, nb_out_channels=1, *args, **kwargs):
        self._binded_conv_layer = binded_conv_layer
        self.nb_out_channels = nb_out_channels
        kwargs['nb_filter'] = self._binded_conv_layer.nb_filter
        kwargs['nb_row'] = self._binded_conv_layer.nb_row
        kwargs['nb_col'] = self._binded_conv_layer.nb_col
        super(Deconvolution2D, self).__init__(*args, **kwargs)

    def build(self):
        self.W = self._binded_conv_layer.W.dimshuffle((1, 0, 2, 3))
        if self.dim_ordering == 'th':
            self.W_shape = (self.nb_out_channels, self.nb_filter, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            raise NotImplementedError()
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.b = K.zeros((self.nb_out_channels,))
        self.params = [self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        output_shape = list(super().output_shape)

        if self.dim_ordering == 'th':
            output_shape[1] = self.nb_out_channels
        elif self.dim_ordering == 'tf':
            output_shape[0] = self.nb_out_channels
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return tuple(output_shape)

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = deconv2d_fast(X, self.W,
                                 strides=self.subsample,
                                 border_mode=self.border_mode,
                                 dim_ordering=self.dim_ordering,
                                 image_shape=self.input_shape,
                                 filter_shape=self.W_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_out_channels, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_out_channels))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DePool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self._pool2d_layer.get_output(train)), wrt=self._pool2d_layer.get_input(train)) * output

        return f


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
			a = a.replace('.txt', '.jpg')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)


print len(file_list)


#1024x768





#number = 100

data = []
neighbors = []
groundtruth = []
counter = 0
counts = []
count1 = 0
if (_platform == "darwin") : 
	seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_GroundTruth'
else :
	seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'

		
for fn in file_list :
	if (fn.endswith('.jpg')) :
		if (number == None or counter < number) : 

			# This section creates the feature vector to feed to the network. It reads the RGB value of
			# each superpixel and adds it to a list. 


			print 'loading ' + fn

			fn2 = fn.replace('.jpg', '_Segmentation.png')

			im = Image.open(fn)

			#width, height = im.size

			width, height = im.size
			img = im.resize((768, 768), Image.ANTIALIAS)

			width2, height2 = img.size

			if width==width2 and height2== height:
				print "HSDJKAFHALSKDH"

			temp = np.array(img)

			data.append(temp)

			

			if (_platform == "darwin") : 
				long_fn = seg_gt_dir + "/" + fn2
			else :
				long_fn = seg_gt_dir + '\\' + fn2

			ground_file = Image.open(long_fn)

			tground = ground_file.resize((768, 768),Image.ANTIALIAS)

			tempground = np.array(tground)

			tempground = tempground/255.0

			groundtruth.append(tempground)
			counter += 1

			# This section reads the groundtruth file




input_img = Input(shape=(3, 768, 768))

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((4, 4), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((4, 4), border_mode='same')(x)
x = Convolution2D(64, 6, 6, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(64, 6, 6, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)

x = UpSampling2D((4, 4))(x)

x = Convolution2D(18, 3, 3, activation='relu', border_mode='same')(x)

x = UpSampling2D((2, 2))(x)

#x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)

#x = UpSampling2D((2, 2))(x)

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)

x = UpSampling2D((4, 4))(x)

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)

x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
print autoencoder.summary()
data = np.asarray(data)
data = data.reshape(data.shape[0], 3, 768, 768)
groundtruth = np.asarray(groundtruth)
groundtruth = groundtruth.reshape(groundtruth.shape[0], 1, 768, 768)

autoencoder.fit(np.asarray(data), np.asarray(groundtruth), nb_epoch=2, batch_size=1, validation_split=0.1)
scores = autoencoder.evaluate(np.asarray(data), np.asarray(groundtruth))
#print("%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))



# This part saves the network
model_json = autoencoder.to_json()
with open("convnetwork24.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
autoencoder.save_weights("convnetwork24.h5")
print("Saved model to disk, 24")







