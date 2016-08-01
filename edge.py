import numpy as np
import os
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.io import imread
from PIL import Image
from skimage.segmentation import slic
from math import sqrt
import collections
from collections import defaultdict
import argparse
from sys import platform as _platform
import json
from pprint import pprint
import csv
import matplotlib.pyplot as plt
import sys


ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')
args = vars(ap.parse_args())

file_list =[]




if (args['list'] != None) :
	with open(args['list']) as batch_file :
		for line in batch_file :
			if line.endswith('.txt') :
				line = line.replace('.txt', '.jpg')			
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)

print file_list

for fn in file_list :
	if (fn.endswith('.txt')) :
		counter = 0

		input_file = csv.DictReader(open(fn))
		print "Processing " + fn
		#print input_file.fieldnames
		#fn2 = fn.replace('.txt', '.jpg')
		fn3 = fn.replace('_Segmentation.txt', '_neighbors.txt')


		orig_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_Data/'
		fn_f = orig_dir + fn3

		neighbors_file = csv.DictReader(open(fn_f))

		vals = [input_file.fieldnames[2]]


		for row in input_file:
			label = float(row["label"])
			val = float(row[" mask0"])
			vals.append(val)

		finalval = []




		for row in neighbors_file:
			curr = int(row["label"])
			north = int(row[" north"])
			south = int(row[" south"])
			east = int(row[" east"])
			west = int(row[" west"])

			southeast = int(row[" southeast"])
			southwest = int(row[" southwest"])
			northeast = int(row[" northeast"])
			northwest = int(row[" northwest"])

			check = int(vals[curr])

			edge = False
			print curr

			if (north!=-1 and curr!=0 and north!=0) :
				if (vals[north] != check) :
					print check, vals[north], north, "north"
					edge = True


			if (south!=-1 and curr!=0 and south!=0):
				if (vals[south] != check) :
					print check, vals[south],south, "south", vals[south] != 0.0
					edge = True

			if (east!=-1 and curr!=0 and east!=0):
				if( vals[east] != check) :
					print check, vals[east], east, "east"
					edge = True

			if (west!=-1 and curr!=0 and west!=0):
				if (vals[west] != check) :
					print check, vals[west], west, "west"
					edge = True

			if (southeast!=-1 and vals[southeast] != check and curr!=0 and southeast!=0) :
				print check, vals[southeast], "southeast"
				edge = True

			if (southwest!=-1 and vals[southwest] != check and curr!=0 and southwest!=0) :
				print check, vals[southwest], "southwest"
				edge = True

			if (northeast!=-1 and vals[northeast] != check and curr!=0 and northeast!=0) :
				print check, vals[northeast], "northeast"
				edge = True

			if (northwest!=-1 and vals[northwest] != check and curr!=0 and northwest!=0) :
				print check, vals[northwest], "northwest"
				edge = True

			if (edge) :
				finalval.append(1)
			else :
				finalval.append(0)
			counter +=1



		os.remove(fn)

		f_out = open(fn, 'w')
		dict_str = ('label, mask' + '\n')

		for i in range(counter) :
			dict_str += str(i) + ', ' + str(finalval[i]) + '\n'

		f_out.write(dict_str)
		f_out.close()



















