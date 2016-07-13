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
        imarr_orig = np.array(original)

        img_row = len(imarr_orig)
        img_col = len(imarr_orig[0])

        txt_fn = fn.replace('.jpg', '.txt')
        counter = -1
        txt_file = open(txt_fn)
        for line in txt_file :
            counter +=1
        txt_file.close()

        feature_arr = [[0 for i in range(15)] for j in range(counter)]
        half_diag = (sqrt((img_row**2) + (img_col**2)))/2

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
                feature_arr[sp][12] = (sqrt(abs(feature_arr[sp][1] - (img_row/2))**2 + abs(feature_arr[sp][2] - (img_col/2))**2 ))/half_diag
                feature_arr[sp][13] = float(feature_arr[sp][1]/img_row)
                feature_arr[sp][14] = float(feature_arr[sp][2]/img_col)

        os.remove(txt_fn)

        f_out = open(txt_fn, 'w')
        dict_str = ('Superpixel label, Centroid row, Centroid column, Area,'
                + ' Avg R value, Avg G value, Avg B value, Dissimilarity, Correlation,'
                + ' Contrast, Energy, Homogeneity, Distance from center, Normalized row, Normalized column' + '\n')

        for i in range(counter) :
            dict_str += (str(int(feature_arr[i][0])) + ', ' 
                + str(int(feature_arr[i][1])) + ', ' + str(int(feature_arr[i][2]))
                 + ', ' + str(feature_arr[i][3])
                  + ', ' + str(feature_arr[i][4])+ ', ' + str(feature_arr[i][5]) + ', ' 
                  + str(feature_arr[i][6]) + ', ' + str(feature_arr[i][7]) + ', ' 
                  + str(feature_arr[i][8]) + ', ' + str(feature_arr[i][9])  + ', ' + str(feature_arr[i][10]) + 
                  ', ' + str(feature_arr[i][11]) + ', ' + str(feature_arr[i][12]) + ', '
                  + str(feature_arr[i][13]) + ', ' + str(feature_arr[i][14]) + '\n')

        f_out.write(dict_str)
        f_out.close()
