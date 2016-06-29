# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from collections import defaultdict
import matplotlib.pyplot as plt

from scipy import misc
#import scipy.misc.pilutil as smp
import scipy.ndimage
import numpy as np
import argparse
from PIL import Image
import cv2
import collections
import csv
 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-j", "--mask", required = True, help = "Path to mask")
args = vars(ap.parse_args())
args2 = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
maskimage = img_as_float(io.imread(args2["mask"]))

#greyimage = Image.open(args["image"]).convert('LA')
#agreyimage = np.asarray(greyimage)


gray = rgb2gray(image)    

 
# loop over the number of segments
#for numSegments in (100, 200, 300):
#for i in range of segements.shape of zero
	# apply SLIC and extract (approximately) the supplied number
	# of segments
segments = slic(image, n_segments = 1000, sigma = 5, slic_zero = 2)
 
	# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (1000))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
#fig.savefig('/Users/18AkhilA/Desktop/superpix.jpg')
plt.axis("off")
 
# show the plots
plt.savefig('superpix.png', bbox_inches='tight')
plt.show()

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

dict = collections.Counter()
dict2 = collections.Counter()
dict3 = collections.Counter()
dict4 = collections.Counter()
dict5 = collections.Counter()

xr = len(image)
        #print xr
yr = len(image[0])
        #print yr

for i in range(0, xr):
    for j in range(0, yr): 
                #print i
                #print j
        pixel = image[i, j]
        lbl = segments[i, j]

        dict2[lbl] += 1
        dict[lbl] += pixel
        dict5[lbl] += maskimage[i,j]
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
            dict4[lbl] += 1


maskdict = collections.Counter()

for i in range(0, len(dict5)):
    if (dict5[i] / dict2[i] > 0.5):
        maskdict[i] = 1
        






w = csv.writer(open("output.csv", "w"))
for key, val in dict.items():
    w.writerow([key, val[0]/(dict2[key]), val[1]/(dict2[key]) , val[2]/(dict2[key]), dict3[key]/dict4[key], dict2[key], maskdict[key]] )

















        # show the masked region
        #cv2.imshow("Mask", mask)
       # cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
        #cv2.waitKey(0)

file.close()