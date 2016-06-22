# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse
#import cv2
 
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
counter = 0;

for (i, segVal) in enumerate(np.unique(segments)):
    # construct a mask for the segment
    #print "[x] inspecting segment %d" % (i)
        file.write(str(i) + " ")
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        file.write(str(len(mask[segments == segVal])) + "\n")
        # show the masked region
        #cv2.imshow("Mask", mask)
        #cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
#cv2.waitKey(0)

file.close()