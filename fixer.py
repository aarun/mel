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
       

        






        with open (txt_fn) as csvfile :
            reader = csv.DictReader(csvfile)
            for row in reader :
                sp = (int)(row["Label"])
                feature_arr[sp][0] = int(row["Label"])
                feature_arr[sp][1] = float(row[" Row"])
                feature_arr[sp][2] = float(row[" Column"])
                feature_arr[sp][3] = float(row[" Area"])
                #feature_arr[sp][4] = float(row[" Avg R value"])
                #feature_arr[sp][5] = float(row[" Avg G value"])
                #feature_arr[sp][6] = float(row[" Avg B value"])

                feature_arr[sp][4] = float(row[" R0"])
                feature_arr[sp][5] = float(row[" R1"])
                feature_arr[sp][6] = float(row[" R2"])
                feature_arr[sp][7] = float(row[" R3"])
                feature_arr[sp][8] = float(row[" R4"])
                feature_arr[sp][9] = float(row[" R5"])
                feature_arr[sp][10] = float(row[" R6"])
                feature_arr[sp][11] = float(row[" R7"])

                feature_arr[sp][12] = float(row[" G0"])
                feature_arr[sp][13] = float(row[" G1"])

                temp = str(row[" G2"])
                ind = temp.index(".")

                g2 = temp[:ind]
                g3 = temp[ind:]

                find = g3.index(".", ind)

                find = find-1

                g2 += g3[:find]
                g3 = g3[find:]

                print temp, ind, find, g2, g3 


                feature_arr[sp][14] = float(g2)
                feature_arr[sp][15] = float(g3)
                feature_arr[sp][16] = float(row[" G3"])
                feature_arr[sp][17] = float(row[" G4"])
                feature_arr[sp][18] = float(row[" G5"])
                feature_arr[sp][19] = float(row[" G6"])

                feature_arr[sp][20] = float(row[" G7"])
                feature_arr[sp][21] = float(row[" B0"])
                feature_arr[sp][22] = float(row[" B1"])
                feature_arr[sp][23] = float(row[" B2"])
                feature_arr[sp][24] = float(row[" B3"])
                feature_arr[sp][25] = float(row[" B4"])
                feature_arr[sp][26] = float(row[" B5"])
                feature_arr[sp][27] = float(row[" B6"])
                feature_arr[sp][28] = float(row[" B7"])


                feature_arr[sp][29] = float(row[" Dissimilarity"])
                feature_arr[sp][30] = float(row[" Correlation"])
                feature_arr[sp][31] = float(row[" Contrast"])
                feature_arr[sp][32] = float(row[" Energy"])
                feature_arr[sp][33] = float(row[" Homogeneity"])
                feature_arr[sp][34] = float(row[" Distance_center"])
                feature_arr[sp][35] = float(row[" Norm_row"])
                


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
