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


	#print args

	data = []
	groundtruth = []
	counter = 0
	if (_platform == "darwin") : 
		seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_GroundTruth'
	else :
		seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'
	for fn in file_list :
		if (fn.endswith('.txt')) :
			if (number != None) : 
				if (counter < number) :

					

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

					data.extend(temp)

					fn2 = fn.replace('.txt', '_Segmentation.txt')

					if (_platform == "darwin") : 
						long_fn = seg_gt_dir + "/" + fn2
					else :
						long_fn = seg_gt_dir + '\\' + fn2

					ground_file = csv.DictReader(open(long_fn))

					#print ground_file.fieldnames

					m = [0]

					for row in ground_file:
						m.append(int(row[" mask0"]))

					groundtruth.extend(m)
					counter += 1

				



			else :
				
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

				data.extend(temp)

				fn2 = fn.replace('.txt', '_Segmentation.txt')
				if (_platform == "darwin") : 
						long_fn = seg_gt_dir + "/" + fn2
				else :
					long_fn = seg_gt_dir + '\\' + fn2

				ground_file = csv.DictReader(open(long_fn))

				#print ground_file.fieldnames

				m = [0]

				for row in ground_file:
					m.append(int(row[" mask0"]))

				groundtruth.extend(m)


					



				#print fn
				#temp = np.loadtxt(fn, delimiter= ",", skiprows=1)

				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#data.append(temp)
				#fn2 = fn.replace('.txt', '_Segmentation.txt')
				#long_fn = seg_gt_dir + "/" + fn2
				
				#grnd = np.loadtxt(long_fn, dtype = int,delimiter= ',', skiprows=1, usecols=[0])

				#grnd = np.delete(grnd, (0), axis = 1)
				#groundtruth.append(grnd)

else :

	data = []
	groundtruth = []
	counter = 0
	if (_platform == "darwin") : 
		seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_GroundTruth'
	else :
		seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'
	for fn in os.listdir('.') :
		if (fn.endswith('.txt')) :
			if (number != None) : 
				if (counter < number) :

					

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

					data.extend(temp)

					fn2 = fn.replace('.txt', '_Segmentation.txt')
					if (_platform == "darwin") : 
						long_fn = seg_gt_dir + "/" + fn2
					else :
						long_fn = seg_gt_dir + '\\' + fn2

					ground_file = csv.DictReader(open(long_fn))

					#print ground_file.fieldnames

					m = [0]

					for row in ground_file:
						m.append(int(row[" mask0"]))

					groundtruth.extend(m)
					counter += 1

				



			else :
				
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

				data.extend(temp)

				fn2 = fn.replace('.txt', '_Segmentation.txt')
				if (_platform == "darwin") : 
						long_fn = seg_gt_dir + "/" + fn2
				else :
					long_fn = seg_gt_dir + '\\' + fn2

				ground_file = csv.DictReader(open(long_fn))

				#print ground_file.fieldnames

				m = [0]

				for row in ground_file:
					m.append(int(row[" mask0"]))

				groundtruth.extend(m)


					



				#print fn
				#temp = np.loadtxt(fn, delimiter= ",", skiprows=1)

				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#temp = np.delete(temp, (0), axis = 1)
				#data.append(temp)
				#fn2 = fn.replace('.txt', '_Segmentation.txt')
				#long_fn = seg_gt_dir + "/" + fn2
				
				#grnd = np.loadtxt(long_fn, dtype = int,delimiter= ',', skiprows=1, usecols=[0])

				#grnd = np.delete(grnd, (0), axis = 1)
				#groundtruth.append(grnd)
			
			

print 'TRAINING'
forest = RandomForestRegressor(n_estimators = 500, n_jobs = 1)
forest.fit(data, groundtruth)

#joblib.dump(forest, 'forest.pkl')

with open('/Users/18AkhilA/Documents/mel/forest.pkl', 'wb') as f:
    cPickle.dump(forest, f)












