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
path = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training/'

if (args['list'] != None) :
	with open(args['list']) as batch_file :
		for line in batch_file :
			if line.endswith('.jpg\n') :
				line = line.replace('.jpg', '.txt')			
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)



for fn in file_list :
	if (fn.endswith('.txt')) :
		
		print "here"

		input_file = csv.DictReader(open(fn))
		print "Predicting " + fn
		#print input_file.fieldnames
		fn2 = fn.replace('.txt', '.jpg')
		original = Image.open(fn2)
		imarr_orig = np.array(original)
		segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)

		fig = plt.figure("Superpixels -- %d segments" % (5000))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(original, segments))
		#fig.savefig('/Users/18AkhilA/Desktop/superpix.jpg')
		plt.axis("off")
 
		# show the plots
		plt.savefig('superpix.png', bbox_inches='tight')
		plt.show()



		graph = np.empty((70, 70), dtype=np.int)
		graph[:] = -1
		print graph

		cx = 0
		cy = 0

		count = 0
		nextlabel = 0
		check = [0]
		


		


		while (len(check) < len(np.unique(segments)) ) : 

			mask2 = np.zeros(segments.shape[:2], dtype='uint8')
			mask2[segments == nextlabel] = 255
			area = len(mask2[segments == nextlabel])	
			sp_locations = mask2[:,:] == 255
			props = regionprops(mask2, cache=True )

			graph[cx][cy] = nextlabel

			olbl = nextlabel
			

			currlbl = nextlabel
			
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


			if (x < len(imarr_orig) and graph[cx+1][cy] == -1) :
				
				graph[cx+1][cy] = currlbl
				down = True

			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and x < len(imarr_orig) and x >= 0:
				currlbl = segments[x][y]
				x -=1
			if (x < len(imarr_orig) and x > 0 and graph[cx-1][cy] == -1) :
				
				graph[cx-1][cy] = currlbl
				up = True


			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and y < len(imarr_orig[0]) and y >= 0:
				currlbl = segments[x][y]
				y +=1
			if (y < len(imarr_orig[0]) and graph[cx][cy+1] == -1) :
				
				graph[cx][cy+1] = currlbl
				right = True


			x = (int)(props[0].centroid[0])
			y = (int)(props[0].centroid[1])
			currlbl = olbl

			while currlbl == olbl and y < len(imarr_orig[0]) and y > 0:
				currlbl = segments[x][y]
				y -=1
			if (y < len(imarr_orig[0]) and y > 0 and graph[cx][cy-1] == -1) :
				
				graph[cx][cy-1] = currlbl
				left = True

			if not (olbl in check):
				check.append(olbl)


			

			if ( cy < len(graph[0])-1 and graph[cx][cy+1] == -1) :
				right = False
			if (cx < len(graph)-1 and graph[cx+1][cy] == -1) :
				down = False
			if (cy > 0 and graph[cx][cy-1] == -1) :
				left = False
			if (cx > 0 and graph[cx-1][cy] == -1) :
				up = False




			if right and graph[cx][cy+1] != -1 and not (graph[cx][cy+1] in check) :
				cy = cy+1
				nextlabel = graph[cx][cy]
				print "right"
			elif down and graph[cx+1][cy] != -1 and not (graph[cx+1][cy] in check):
				cx = cx+1
				nextlabel = graph[cx][cy]
				print "down"
			elif left and graph[cx][cy-1] != -1 and not (graph[cx][cy-1] in check):
				cy = cy-1
				nextlabel = graph[cx][cy]
				print "left"
			elif up and graph[cx-1][cy] != -1 and not (graph[cx-1][cy] in check):
				cx = cx-1
				nextlabel = graph[cx][cy]
			else :
				cx = 0
				cy = 0
				while (graph[cx][cy] in check) :
					cy +=1
					if (cy >= len(graph[0])) :
						cy = 0
						cx += 1
				if (graph[cx][cy] == -1) :
					
					break
				
				nextlabel = graph[cx][cy]





		graph = graph[~np.all(graph == -1, axis=1)]
		graph = graph[:, ~np.all(graph == -1, axis=0)]

		


		f_out = (open('grid.txt','w'))

		graph_s = ('grid' + '\n')

		for k in range(len(graph)) :
			graph_s += (str(graph[k]) + '\n')


		f_out.write(graph_s)
		f_out.close()






