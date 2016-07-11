import numpy as np
from skimage.segmentation import slic
from PIL import Image
from sys import platform as _platform
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())


file_list =[]

with open(args['list']) as batch_file :
	for line in batch_file :
		if line.endswith('.jpg\n') :
			line = line.replace('.jpg', '.txt')
		a = line.strip('\n')
		file_list.append(a)

#print file_list

#stnd = np.loadtxt("output6.csv", delimiter= ",")
print 'image              , accuracy, sensitivity, specificity, dice_coeff, jaccard_ind'
for fn in file_list:
    if fn.endswith('.txt'):


		root_name = fn.strip('.txt')
		#print 'Processing file root: ', root_name		
		prediction = np.loadtxt(root_name+'_Prediction.csv', delimiter= ",")

		falsepos = 0
		falseneg = 0
		trueneg = 0
		truepos = 0

		if (_platform == "darwin") : 
			seg_gt_dir = '/users/sahana/mel/ISBI2016_ISIC_Part1_Training_GroundTruth'
			orig_dir = '/Users/sahana/Mel/ISBI2016_ISIC_Part1_Training_Data/'
			gt_fn = seg_gt_dir + "/" + root_name + '_Segmentation.png'			
		else :
			seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_GroundTruth'
			orig_dir = 'C:\mel\ISBI2016_ISIC_Part1_Training_Data\\'			
			gt_fn = seg_gt_dir + "\\" + root_name + '_Segmentation.png'			

		gt = Image.open(gt_fn,'r')
		original = Image.open(orig_dir + root_name + '.jpg')
		imarr_gt = np.array(gt)
		imarr_image = np.array(original)
		imarr_predict = np.zeros((imarr_image.shape[0], imarr_image.shape[1]))
		segments = slic(original, n_segments = 3000, sigma = 5, slic_zero = 2)
		imarr_overlay = np.array(original)

		for (i, segVal) in enumerate(np.unique(segments)) :

				val = prediction[segVal]

				if (val < 0.5) :
					imarr_predict[segments == segVal] = 0
				else :
					imarr_predict[segments == segVal] = 255


		for i in range(len(imarr_gt)) :
			for j in range(len(imarr_gt[i])) :
				if (imarr_gt[i][j] == 255):
					if (imarr_predict[i][j] == 0) :
						falseneg += 1
						imarr_overlay[i][j] = [255,0,0]
					else :
						truepos += 1
						imarr_overlay[i][j] = [255,255,255]
				else :
					if (imarr_predict[i][j] == 0) :
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

		print root_name, accuracy, sensitivity, specificity, dice_coeff, jaccard_ind
		#print accuracy
		#print sensitivity
		#print specificity
		#print dice_coeff
		#print jaccard_ind




