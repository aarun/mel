import numpy as np
import os
from skimage.measure import regionprops
from skimage import data
from skimage.io import imread
from PIL import Image
from skimage.segmentation import slic
from math import sqrt
import collections
from collections import defaultdict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())


file_list =[]

with open(args['list']) as batch_file :
    for line in batch_file :
        a = line.strip('\n')
        file_list.append(a)

print file_list

for fn in file_list:
    if fn.endswith('jpg') :

        txt_fn = fn.replace('jpg', 'txt')
        w = csv.writer(open(txt_fn, "w"))
        for line in txt_fn :
            w.append(line)

        print 'Processing file: ', fn
        original = Image.open(fn)
        segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)
        sp_dict = {}

        xr = len(imarr_orig)
        yr = len(imarr_orig[0])
        
        for (i, segVal) in enumerate(np.unique(segments)) :
            mask2 = np.zeros(segments.shape[:2], dtype='uint8')
            mask2[segments == segVal] = 255
            props = regionprops(mask2, cache=True )

            avg = (xr + yr)/2
            distance = (sqrt(abs(props[0].centroid[0] - (xr/2))**2 + abs(props[0].centroid[1] - (yr/2))**2 ))/avg

            sp_dict[segVal] = distance
        
        dict_str = ('Distance from center')

        counter = -1
        for row in w :
            if counter !=-1 :
                row[12] = sp_dict[counter]
            counter +=1

        f_out.write(dict_str)
        f_out.close()
