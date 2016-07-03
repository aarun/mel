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


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-j", "--mask", required = True, help = "Path to mask")
args = vars(ap.parse_args())
args2 = vars(ap.parse_args())


image = img_as_float(io.imread(args["image"]))

maskimage = io.imread(args2["mask"])

segments = slic(image, n_segments = 1000, sigma = 5, slic_zero = 2)

data = np.loadtxt("output2.csv", delimiter= ",")
#print(data[40][])

data2 = np.zeros( (len(image),len(image[0])), dtype=np.uint8 )

xr = len(image)
#print xr
yr = len(image[0])
#print yr
#print(segments)



for i in range(0, xr):
    for j in range(0, yr): 
    	lbl = segments[i][j]
    	#print(lbl)
    	val = data[lbl]
    	#print(val)
    	if (val == 2):
    		data2[i][j] = 1
    	else:
    		data2[i][j] = 0
    	#data2[i][j] = val




#for (i, segVal) in enumerate(np.unique(segments)):
    # construct a mask for the segment
    #print "[x] inspecting segment %d" % (i)
       # file.write(str(i) + " ")
       # mask = np.zeros(image.shape[:2], dtype = "uint8")
       # mask[segments == segVal] = data[segVal] 
        #print(mask[segments == segVal])
        #print(data[segVal])
       # file.write(str(len(mask[segments == segVal])) + "\n")

        
       # pix = mask
       # pixs = pix[segments == segVal]

        #img = misc.toimage(pixs)
        #image2 = Image.fromarray(pixs)
        #colors = image2.getcolors(maxcolors=256)

fig = plt.figure("Superpixels -- %d segments" % (5000))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, data2))
#fig.savefig('/Users/18AkhilA/Desktop/superpix.jpg')
plt.axis("off")
 
# show the plots
plt.show()


print np.unique(data2)
data2 = data2.astype('uint8')*255 

im = Image.fromarray(data2)

im.show()








