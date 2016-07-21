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

        feature_arr = [[0 for i in range(36)] for j in range(counter)]
        segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)

        print "sliced"

        rhist = []
        ghist = []
        bhist = []



        for (i, segVal) in enumerate(np.unique(segments)) :   

            mask2 = np.zeros(segments.shape[:2], dtype='uint8')
            mask2[segments == segVal] = 255
            area = len(mask2[segments == segVal])   
            sp_locations = mask2[:,:] == 255

            r, redge = np.histogram(imarr_orig[sp_locations,0], bins = 8, range = (0, 255))
            g, gedge = np.histogram(imarr_orig[sp_locations,1], bins = 8, range = (0, 255))
            b, bedge = np.histogram(imarr_orig[sp_locations,2], bins = 8, range = (0, 255))

            rhist.append(r)
            ghist.append(g)
            bhist.append(b)

        #print rhist
        #print ghist
        #print bhist






        with open (txt_fn) as csvfile :
            reader = csv.DictReader(csvfile)
            for row in reader :
                sp = (int)(row["Superpixel label"])
                feature_arr[sp][0] = int(row["Superpixel label"])
                feature_arr[sp][1] = float(row[" Centroid row"])
                feature_arr[sp][2] = float(row[" Centroid column"])
                feature_arr[sp][3] = float(row[" Area"])
                #feature_arr[sp][4] = float(row[" Avg R value"])
                #feature_arr[sp][5] = float(row[" Avg G value"])
                #feature_arr[sp][6] = float(row[" Avg B value"])
                feature_arr[sp][4] = float(rhist[sp][0]/feature_arr[sp][3])
                feature_arr[sp][5] = float(rhist[sp][1]/feature_arr[sp][3])
                feature_arr[sp][6] = float(rhist[sp][2]/feature_arr[sp][3])
                feature_arr[sp][7] = float(rhist[sp][3]/feature_arr[sp][3])
                feature_arr[sp][8] = float(rhist[sp][4]/feature_arr[sp][3])
                feature_arr[sp][9] = float(rhist[sp][5]/feature_arr[sp][3])
                feature_arr[sp][10] = float(rhist[sp][6]/feature_arr[sp][3])
                feature_arr[sp][11] = float(rhist[sp][7]/feature_arr[sp][3])

                feature_arr[sp][12] = float(ghist[sp][0]/feature_arr[sp][3])
                feature_arr[sp][13] = float(ghist[sp][1]/feature_arr[sp][3])
                feature_arr[sp][14] = float(ghist[sp][2]/feature_arr[sp][3])
                feature_arr[sp][15] = float(ghist[sp][3]/feature_arr[sp][3])
                feature_arr[sp][16] = float(ghist[sp][4]/feature_arr[sp][3])
                feature_arr[sp][17] = float(ghist[sp][5]/feature_arr[sp][3])
                feature_arr[sp][18] = float(ghist[sp][6]/feature_arr[sp][3])
                feature_arr[sp][19] = float(ghist[sp][7]/feature_arr[sp][3])

                feature_arr[sp][20] = float(bhist[sp][0]/feature_arr[sp][3])
                feature_arr[sp][21] = float(bhist[sp][1]/feature_arr[sp][3])
                feature_arr[sp][22] = float(bhist[sp][2]/feature_arr[sp][3])
                feature_arr[sp][23] = float(bhist[sp][3]/feature_arr[sp][3])
                feature_arr[sp][24] = float(bhist[sp][4]/feature_arr[sp][3])
                feature_arr[sp][25] = float(bhist[sp][5]/feature_arr[sp][3])
                feature_arr[sp][26] = float(bhist[sp][6]/feature_arr[sp][3])
                feature_arr[sp][27] = float(bhist[sp][7]/feature_arr[sp][3])


                feature_arr[sp][28] = float(row[" Dissimilarity"])
                feature_arr[sp][29] = float(row[" Correlation"])
                feature_arr[sp][30] = float(row[" Contrast"])
                feature_arr[sp][31] = float(row[" Energy"])
                feature_arr[sp][32] = float(row[" Homogeneity"])
                feature_arr[sp][33] = float(row[" Distance from center"])
                feature_arr[sp][34] = float(row[" Normalized row"])
                feature_arr[sp][35] = float(row[" Normalized column"])


        os.remove(txt_fn)

        f_out = open(txt_fn, 'w')
        dict_str = ('Label, Row, Column, Area,'
                + ' R0, R1, R2, R3, R4, R5, R6, R7,'
                + ' G0, G1, G2, G3, G4, G5, G6, G7,'
                + ' B0, B1, B2, B3, B4, B5, B6, B7,'
                + ' Dissimilarity, Correlation,'
                + ' Contrast, Energy, Homogeneity, Distance_center, Norm_row, Norm_column' + '\n')




        for i in range(counter) :
            
            dict_str += (str(int(feature_arr[i][0])) + ', ' 
                + str(int(feature_arr[i][1])) + ', ' + str(int(feature_arr[i][2]))
                 + ', ' + str(feature_arr[i][3])
                  + ', ' + str(feature_arr[i][4])+ ', ' + str(feature_arr[i][5]) + ', ' 
                  + str(feature_arr[i][6]) + ', ' + str(feature_arr[i][7]) + ', ' 
                  + str(feature_arr[i][8]) + ', ' + str(feature_arr[i][9])  + ', ' + str(feature_arr[i][10]) + 
                  ', ' + str(feature_arr[i][11]) + ', ' + str(feature_arr[i][12]) + ', '
                  + str(feature_arr[i][13]) + ', ' + str(feature_arr[i][14]) + ', ' + str(feature_arr[i][15]) + 
                  ', ' + str(feature_arr[i][16]) + ', ' + str(feature_arr[i][17])+', ' + str(feature_arr[i][18])+
                  ', ' + str(feature_arr[i][19]) + ', ' + str(feature_arr[i][20]) + ', ' + str(feature_arr[i][21]) +
                  ', ' + str(feature_arr[i][22]) + ', ' + str(feature_arr[i][23]) + ', ' + str(feature_arr[i][24]) 
                  + ', ' + str(feature_arr[i][25]) + ', ' + str(feature_arr[i][26]) + ', ' + str(feature_arr[i][27])
                   + ', ' + str(feature_arr[i][28]) + ', ' + str(feature_arr[i][29]) + ', ' + str(feature_arr[i][30])
                   + ', ' + str(feature_arr[i][31]) + ', ' + str(feature_arr[i][32]) + ', ' + str(feature_arr[i][33])
                   + ', ' + str(feature_arr[i][34]) + ', ' + str(feature_arr[i][35]) + '\n')

        f_out.write(dict_str)
        #print dict_str
        f_out.close()
