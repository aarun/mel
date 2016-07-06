# import the necessary packages
from __future__ import division
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from collections import defaultdict
import matplotlib.pyplot as plt
import imutils
from scipy import misc
#import scipy.misc.pilutil as smp
import scipy.ndimage
import numpy as np
import argparse
from PIL import Image
import cv2
import collections
import csv
from math import hypot
from math import sqrt
import os
 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#direct1 = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_Data'
#direct2 = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Training_GroundTruth'


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-j", "--mask", required = True, help = "Path to mask")
args = vars(ap.parse_args())
args2 = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
maskimage = img_as_float(io.imread(args2["mask"]))

greyimage = Image.open(args["image"]).convert('LA')
agreyimage = np.asarray(greyimage)


gray = rgb2gray(image)    

 
# loop over the number of segments
#for numSegments in (100, 200, 300):
#for i in range of segements.shape of zero
	# apply SLIC and extract (approximately) the supplied number
	# of segments
segments = slic(image, n_segments = 3000, sigma = 5, slic_zero = 2)
 
	# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (5000))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
#fig.savefig('/Users/18AkhilA/Desktop/superpix.jpg')
plt.axis("off")
 
# show the plots
plt.savefig('superpix.png', bbox_inches='tight')
#plt.show()

file = open("supersize.txt", "w")
counter = 0




#for (i, segVal) in enumerate(np.unique(segments)):
    # construct a mask for the segment
    #print "[x] inspecting segment %d" % (i)
       # file.write(str(i) + " ")
       # mask = np.zeros(image.shape[:2], dtype = "uint8")
       # mask[segments == segVal] = 255
       # file.write(str(len(mask[segments == segVal])) + "\n")

        
       # pix = mask
       # pixs = pix[segments == segVal]

        #img = misc.toimage(pixs)
        #image2 = Image.fromarray(pixs)
        #colors = image2.getcolors(maxcolors=256) 


#get size and average color of each superpixel

dict = collections.Counter() # color
dict2 = collections.Counter() # size
dict3 = collections.Counter() # LBP
dict4 = collections.Counter() # LBP size
dict5 = collections.Counter() # ground truth checker

# these dictionaries are for determining which side each superpixel is closest to
north = collections.Counter()
south = collections.Counter()
east = collections.Counter()
west = collections.Counter()

centerinx = collections.Counter() # distance to center in x direction
centeriny = collections.Counter() # distance to center in y direction
both = {} # distance in both directions


xr = len(image)
        #print xr
yr = len(image[0])

    
centerx = xr/2
centery = yr/2

        #print yr

for i in range(0, xr):
    for j in range(0, yr): 
                #print i
                #print j
        
        #initialize dictionary for distance to center
        pixel = image[i, j]
        lbl = segments[i, j]
        both[lbl] = 0

        dict2[lbl] += 1 # size
        dict[lbl] += pixel # average color
        dict5[lbl] += maskimage[i,j] # determine groundtruth 


        #distance to closest side
        north[lbl] += i
        south[lbl] += xr - i
        east[lbl] += yr - j
        west[lbl] += j


        # distance in x and y to center

        if (i > xr/2):
            centerinx[lbl] += i-(xr/2)
        else:
            centerinx[lbl] += (xr/2)-i

        if (j > yr/2):
            centeriny[lbl] += j - (yr/2)
        else:
            centeriny[lbl] += (yr/2)-j

            #both[lbl] += hypot(i - (xr/2), j - (yr/2))

        both[lbl] += sqrt( abs(i - (xr/2))**2 + abs(j - (yr/2))**2 ) # overall distance to center

            #if (lbl == 49):
              #  print "HELLO"
              #  print sqrt( abs(i - (xr/2))**2 + abs(j - (yr/2))**2 )



        # calculating LBP

        if (i > 0 and j > 0 and i < xr-1 and j < yr-1):
            code = 0
            code |= int(gray[i-1,j-1] > gray[i,j]) << 7
            code |= int(gray[i-1,j] > gray[i,j]) << 6
            code |= int(gray[i-1,j+1] > gray[i,j]) << 5
            code |= int(gray[i,j+1] > gray[i,j]) << 4
            code |= int(gray[i+1,j+1] > gray[i,j]) << 3
            code |= int(gray[i+1,j] > gray[i,j]) << 2
            code |= int(gray[i+1,j-1] > gray[i,j]) << 1
            code |= int(gray[i,j-1] > gray[i,j]) << 0
            dict3[lbl] += code
            dict4[lbl] += 1 # LBP size 

#print centerinx

maskdict = collections.Counter() # groundtruth for each superpixel
directdict = collections.Counter() # 
xandy = collections.Counter()

for i in range(0, len(north)):
    if (dict5[i] / dict2[i] > 0.5):
        maskdict[i] = 1

    if (north[i] < south[i] and north[i] < east[i] and north[i] < west[i] ):
        directdict[i] = (north[i]/dict2[i])
        xandy[i] +=1
    elif (south[i] < north[i] and south[i] < east[i] and south[i] < west[i] ):
        directdict[i] = (south[i]/dict2[i])
        xandy[i] +=1
    elif (east[i] < north[i] and east[i] < south[i] and east[i] < west[i] ):
        directdict[i] = (east[i]/dict2[i])
    else:
        directdict[i] = (west[i]/dict2[i])

#print directdict


#val = {}

#for i in range(len(directdict)):
 #   if (xandy[i] == 1):
  #      directdict[i] = directdict[i]/(xr)
   #     val[i] = directdict[i]/(xr)
    #    #print (directdict[i]/ xr)
   # else:
    #    directdict[i] = directdict[i]/(yr)
     #   val[i] = directdict[i]/(yr)
        #print (directdict[i]/ yr)

    #centerinx[i]  = centerinx[i]/ (xr/2)
    #centeriny[i] = centeriny[i]/ (yr/2)

#print both

# normalizing average distance to center
total = sqrt( (xr/2)**2 + (yr/2)**2 )

for key in both:
    both[key] /= total
        


#print val

#dict = color
#dict2 = size

#dict3 = LBP value
#dict4 = size of lBP
#both = average distance to center of each superpixel

#output 5 and 6 for test


w = csv.writer(open("output.csv", "w"))
t = csv.writer(open("output1.csv", "w"))
for key, val in dict.items():
    w.writerow([val[0]/(dict2[key]), val[1]/(dict2[key]) , val[2]/(dict2[key]),both[key]/(dict2[key])]) #dict3[key]/dict4[key]]) #/(dict2[key]),   centeriny[key]/dict2[key]]) #, centerinx[key]/dict2[key], directdict[key]])  ,dict3[key]/dict4[key]])

for key, val in dict.items():
    t.writerow([maskdict[key]])

















#file.close()