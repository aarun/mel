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
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
 
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

dict = collections.Counter()
dict2 = collections.Counter()


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

print image[0, 0]
print image[60, 60]
print "DONE DONE DONE"

w = csv.writer(open("output.csv", "w"))
for key, val in dict.items():
    w.writerow([key, val[0]/(dict2[key]), val[1]/(dict2[key]) , val[2]/(dict2[key]), dict2[key]] )














        # show the masked region
        #cv2.imshow("Mask", mask)
       # cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
        #cv2.waitKey(0)

file.close()