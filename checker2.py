import numpy as np
from skimage.segmentation import slic
from PIL import Image
from sys import platform as _platform
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = False, help = 'name of batch file')
args = vars(ap.parse_args())


file_list =[]
if (args['list'] != None):
	with open(args['list']) as batch_file :
		for line in batch_file :
			if line.endswith('.jpg\n') :
				line = line.replace('.jpg', '.txt')
			a = line.strip('\n')
			file_list.append(a)
else :
	for fn in os.listdir('.') :
		file_list.append(fn)

#print file_list

#stnd = np.loadtxt("output6.csv", delimiter= ",")
print 'image , accuracy, sensitivity, specificity, dice_coeff, jaccard_ind'
for fn in file_list:
    if fn.endswith('Prediction.jpeg'):


		root_name = fn.strip('_Prediction.jpeg')
		#print 'Processing file root: ', root_name		
		#prediction = np.loadtxt(fn, delimiter= ",")

		falsepos = 0
		falseneg = 0
		trueneg = 0
		truepos = 0

		if (_platform == "darwin") : 
			seg_gt_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Test_GroundTruth'
			orig_dir = '/Users/18AkhilA/Documents/mel/ISBI2016_ISIC_Part1_Test_Data/'
			gt_fn = seg_gt_dir + "/" + root_name + '_Segmentation.png'			
		else :
			seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Test_GroundTruth'
			orig_dir = 'C:\mel\ISBI2016_ISIC_Part1_Test_Data\\'			
			gt_fn = seg_gt_dir + "\\" + root_name + '_Segmentation.png'			

		gt = Image.open(gt_fn,'r')
		original = Image.open(orig_dir + root_name + '.jpg')
		imarr_gt = np.array(gt)
		imarr_image = np.array(original)
		predict = Image.open(fn)
		imarr_predict = np.array(predict)
		imarr_overlay = np.array(original)
		
		for i in range(len(imarr_gt)) :
			for j in range(len(imarr_gt[i])) :
				if (imarr_gt[i][j] == 255):
					if (imarr_predict[i][j][0] == 0) :
						falseneg += 1
						imarr_overlay[i][j] = [255,0,0]
					else :
						truepos += 1
						imarr_overlay[i][j] = [255,255,255]
				else :
					if (imarr_predict[i][j][0] == 0) :
						trueneg += 1
						imarr_overlay[i][j] = [0,0,0]
					else :
						falsepos += 1
						imarr_overlay[i][j] = [0,255,0]

		#print truepos
		#print falseneg
		#print trueneg
		#print falsepos

		tp = float(truepos)
		tn = float(trueneg)
		fp = float(falsepos)
		fn = float(falseneg)

		accuracy = (tp + tn)/(tp + fp + tn + fn)
		sensitivity = tp/(tp + fn)
		specificity = tn/(tn + fp)
		dice_coeff = (2*tp)/((2*tp) + fn + tp)
		jaccard_ind = tp/(tp + fn + fp)

		image = Image.fromarray(imarr_overlay)
		#image.show()
		image.save(root_name + '_error.jpg')	

		print root_name, ',', accuracy, ',', sensitivity, ',', specificity, ',', dice_coeff, ',', jaccard_ind
		#print accuracy
		#print sensitivity
		#print specificity
		#print dice_coeff
		#print jaccard_ind




