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
import csv

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

        print 'Processing file: ', fn
        original = Image.open(fn)
        segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)
        sp_dict = {}

        imarr_orig = np.array(original)

        xr = len(imarr_orig)
        yr = len(imarr_orig[0])

        counter = 0
        
        for (i, segVal) in enumerate(np.unique(segments)) :
            counter += 1
            mask2 = np.zeros(segments.shape[:2], dtype='uint8')
            mask2[segments == segVal] = 255
            props = regionprops(mask2, cache=True )

            half_diag = (sqrt((xr**2) + (yr**2)))/2
            distance = (sqrt(abs(props[0].centroid[0] - (xr/2))**2 + abs(props[0].centroid[1] - (yr/2))**2 ))/half_diag

            sp_dict[segVal] = distance
        
        feature_arr = [[0 for i in range(13)] for j in range(counter)]
    


        txt_fn = fn.replace('.jpg', '.txt')
        #txt_fn2 = fn.replace('.jpg', '_2.txt')
        with open (txt_fn) as csvfile :
            reader = csv.DictReader(csvfile)
            for row in reader :
                sp = (int)(row["Superpixel label"])
                feature_arr[sp][0] = int(row["Superpixel label"])
                feature_arr[sp][1] = float(row[" Centroid row"])
                feature_arr[sp][2] = float(row[" Centroid column"])
                feature_arr[sp][3] = float(row[" Area"])
                feature_arr[sp][4] = float(row[" Avg R value"])
                feature_arr[sp][5] = float(row[" Avg G value"])
                feature_arr[sp][6] = float(row[" Avg B value"])
                feature_arr[sp][7] = float(row[" Dissimilarity"])
                feature_arr[sp][8] = float(row[" Correlation"])
                feature_arr[sp][9] = float(row[" Contrast"])
                feature_arr[sp][10] = float(row[" Energy"])
                feature_arr[sp][11] = float(row[" Homogeneity"])
                feature_arr[sp][12] = float(sp_dict[sp])
                #label += 1

        os.remove(txt_fn)
    
        #with open(txt_fn,'w') as out_csv :
        #    w = csv.writer(out_csv)
        #    w.writerow(str('Superpixel label, Centroid row, Centroid column, Area, Avg R value, Avg G value, Avg B value, Dissimilarity, Correlation, Contrast, Energy, Homogeneity, Distance from center'))
        #   for i in range(counter) :
        #        w.writerow(feature_arr[i])

        f_out = open(txt_fn, 'w')
        dict_str = ('Superpixel label, Centroid row, Centroid column, Area,'
                + ' Avg R value, Avg G value, Avg B value, Dissimilarity, Correlation,'
                + ' Contrast, Energy, Homogeneity, Distance from center' + '\n')

        for i in range(counter) :
            dict_str += (str(int(feature_arr[i][0])) + ', ' 
                + str(int(feature_arr[i][1])) + ', ' + str(int(feature_arr[i][2]))
                 + ', ' + str(feature_arr[i][3])
                  + ', ' + str(feature_arr[i][4])+ ', ' + str(feature_arr[i][5]) + ', ' 
                  + str(feature_arr[i][6]) + ', ' + str(feature_arr[i][7]) + ', ' 
                  + str(feature_arr[i][8]) + ', ' + str(feature_arr[i][9])  + ', ' + str(feature_arr[i][10]) + 
                  ', ' + str(feature_arr[i][11]) + ', ' + str(feature_arr[i][12]) + '\n')

        f_out.write(dict_str)
        #w = csv.writer(csvfile)
        #f = csv.reader(txt_fn)
        #for line in txt_file :
        #    w.writerow(line)


        #counter = -1
        #for row in f :
            #if counter != -1 :
                #try :
                #    row[12] = sp_dict[row[0]]
                #except KeyError :
                #    print row
            #counter +=1

        #counter = 0
        #for line in txt_file :
        #    line = w[counter]
        #    counter +=1
