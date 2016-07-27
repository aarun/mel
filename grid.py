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
		

		input_file = csv.DictReader(open(fn))
		print "Processing " + fn
		#print input_file.fieldnames
		fn2 = fn.replace('.txt', '.jpg')
		fn3 = fn.replace('.txt', '_neighbors.txt')
		original = Image.open(fn2)
		imarr_orig = np.array(original)
		#print imarr_orig
		segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)

		fig = plt.figure("Superpixels -- %d segments" % (5000))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(original, segments))
		#fig.savefig('/Users/18AkhilA/Desktop/superpix.jpg')
		plt.axis("off")
 
		# show the plots
		#plt.savefig('superpix.png', bbox_inches='tight')
		#plt.show()



		graph = []

		cx = 0
		cy = 0

		count = 0
		nextlabel = 0
		check = [0]
		


		


		for (i, segVal) in enumerate(np.unique(segments)) :

			neighbors = [segVal]

			mask2 = np.zeros(segments.shape[:2], dtype='uint8')
			mask2[segments == segVal] = 255
			area = len(mask2[segments == segVal])	
			sp_locations = mask2[:,:] == 255
			props = regionprops(mask2, cache=True )

			

			olbl = segVal
			

			currlbl = segVal
			
			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			
			currlbl = segments[x][y]
			olbl = segments[x][y]

			up = False
			down = False
			right = False
			left = False
			

			while (currlbl == olbl) and (x < len(imarr_orig)) and (x >= 0):
				currlbl = segments[x][y]
				x +=1


			if (x < len(imarr_orig)) :
				
				neighbors.append(currlbl)
				down = True
			else :
				neighbors.append(-1)


			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				x -=1
			if (x < len(imarr_orig) and x > 0) :
				
				neighbors.append(currlbl)
				up = True
			else :
				neighbors.append(-1)


			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and y < len(imarr_orig[0]) and y >= 0:
				currlbl = segments[x][y]
				y +=1
			if (y < len(imarr_orig[0])) :
				
				neighbors.append(currlbl)
				right = True
			else :
				neighbors.append(-1)


			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0:
				currlbl = segments[x][y]
				y -=1
			if (y < len(imarr_orig[0]) and y > 0) :
				
				neighbors.append(currlbl)
				left = True
			else :
				neighbors.append(-1)

			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl



			#bottomright
			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				y +=1
				x +=1
			if (y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0) :
				
				neighbors.append(currlbl)
				left = True
			else :
				neighbors.append(-1)

			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl



			#topleft
			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				y -=1
				x -=1
			if (y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0) :
				
				neighbors.append(currlbl)
				left = True
			else :
				neighbors.append(-1)

			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl



			#top right
			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				x +=1
				y -=1
			if (y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0) :
				
				neighbors.append(currlbl)
				left = True
			else :
				neighbors.append(-1)

			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl


			#bottomleft
			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				x -=1
				y +=1
			if (y < len(imarr_orig[0]) and y > 0 and x < len(imarr_orig) and x >= 0) :
				
				neighbors.append(currlbl)
				left = True
			else :
				neighbors.append(-1)





			graph.append(neighbors)





		


		f_out = (open(fn3,'w'))

		graph_s = ('label, south, north, east, west, southeast, northwest, northeast, southwest' + '\n')

		for k in range(len(graph)) :
			graph_s += (str(graph[k][0]) + ', ' + str(graph[k][1])+ ', '+ str(graph[k][2])+ ', ' + str(graph[k][3]) +', ' + str(graph[k][4]) + 
			', ' + str(graph[k][5]) +', ' + str(graph[k][6]) +', ' + str(graph[k][7]) +', ' + str(graph[k][8]) + '\n')


		f_out.write(graph_s)
		f_out.close()






