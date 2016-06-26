# Test to read Part 2 super pixel images

import numpy as np
import os
from skimage.measure import regionprops

def decodeSuperpixelIndex(rgbValue):
    """
    Decode an RGB representation of a superpixel label into its native scalar value.
    :param pixelValue: A single pixel, or a 3-channel image.
    :type pixelValue: numpy.ndarray of uint8, with a shape [3] or [n, m, 3]
    """
    return \
        (rgbValue[..., 0].astype(np.uint64)) + \
        (rgbValue[..., 1].astype(np.uint64) << np.uint64(8)) + \
        (rgbValue[..., 2].astype(np.uint64) << np.uint64(16))

# This may be used as:
from PIL import Image



#for fn in os.listdir('.') :
    #if fn.endswith('png') :
        #print(fn)
        #f_out=open(fn.replace('.png','.txt'),'w')

fn = 'ISIC_0000000_superpixels.png'
f_out=open(fn.replace('.png','.txt'),'w')

image = Image.open(fn)
assert image.mode == 'RGB'
imarr_enc = np.array(image)
imarr_dec = decodeSuperpixelIndex(imarr_enc)

original = Image.open('ISIC_0000000.jpg')
imarr_orig = np.array(original)

sp_dict = {}

for (i, segVal) in enumerate(np.unique(imarr_dec)) :
#for (i, segVal) in enumerate([0,1]):	
	mask = np.zeros(imarr_dec.shape[:2], dtype='uint8')
	mask[imarr_dec == segVal] = 255
	area = len(mask[imarr_dec == segVal])	
	sp_locations = mask[:,:] == 255


	r = (sum(imarr_orig[sp_locations,0]))/area
	g = (sum(imarr_orig[sp_locations,1]))/area
	b = (sum(imarr_orig[sp_locations,2]))/area

	#print 'Average color: ', r, g, b


	#print " superpixel %d" % (i)
	#print 'Superpixel: ', i, 'Num regions: ', len(props), props[0].centroid, props[0].area
	props = regionprops(mask, cache=True )
	sp_dict[segVal] = [props[0].centroid, area, r, g, b]
	if props[0].area != len(mask[imarr_dec == segVal]):
	    print 'NO'
   

# at this point the sp_dict dictionary - has a list of all the unique superpixel labels
# and also - how many pixels are in each superpixel
#total_pix = 0
#for i in range(len(sp_dict)) :
#    total_pix += sp_dict[i]

#print fn, 'SP: ', imarr_dec.shape, imarr_dec.min(), imarr_dec.max(), imarr_dec.shape[0]*imarr_dec.shape[1]

dict_str = 'Superpixel label, Centroid row, Centroid column, Area, Avg R value, Avg G value, Avg B value' + '\n'
#f_out.write(str(sp_dict))

for k in sp_dict:
    dict_str += (str(k) + ', ' + str(int(sp_dict[k][0][0])) + ', ' + str(int(sp_dict[k][0][1])) + ', ' 
    + str(int(sp_dict[k][1])) + ', ' + str(sp_dict[k][2]) + ', ' + str(sp_dict[k][3]) + ', ' + str(sp_dict[k][4]) + '\n')

f_out.write(dict_str)
f_out.close()

#print sp_dict




