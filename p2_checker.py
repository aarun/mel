import numpy as np
from skimage.segmentation import slic
from PIL import Image
from sys import platform as _platform
import argparse
from sklearn.metrics import roc_auc_score
import csv
import json

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--list', required = True, help = 'name of batch file')
args = vars(ap.parse_args())

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

file_list =[]

with open(args['list']) as batch_file :
	for line in batch_file :
		if line.endswith('.jpg\n') :
			line = line.replace('.jpg', '.txt')
		a = line.strip('\n')
		file_list.append(a)

for fn in file_list:
    if fn.endswith('.txt'):


		root_name = fn.strip('.txt')
		g_prediction = np.loadtxt(root_name+'_Globule_Prediction.csv', delimiter= ",")
		s_prediction = np.loadtxt(root_name + '_Streak_Prediction.csv', delimiter= ",")

		g_falsepos = 0
		g_falseneg = 0
		g_trueneg = 0
		g_truepos = 0

		s_falsepos = 0
		s_falseneg = 0
		s_trueneg = 0
		s_truepos = 0

		if (_platform == "darwin") : 
			seg_gt_dir = '/users/sahana/mel/ISBI2016_ISIC_Part2_Test_GroundTruth'
			orig_dir = '/Users/sahana/Mel/ISBI2016_ISIC_Part2_Test_Data/'
			gt_fn = seg_gt_dir + "/" + root_name + '.json'			
		else :
			seg_gt_dir = 'C:\mel\ISBI2016_ISIC_Part2_Test_GroundTruth'
			orig_dir = 'C:\mel\ISBI2016_ISIC_Part2_Test_Data\\'			
			gt_fn = seg_gt_dir + "\\" + root_name + '.json'			

	
		original = Image.open(orig_dir + root_name + '.jpg')
		g_imarr_overlay = np.array(original)
		s_imarr_overlay = np.array(original)
		super_image = Image.open(orig_dir + root_name + '_superpixels.png')
		imarr_enc = np.array(super_image)
		imarr_dec = decodeSuperpixelIndex(imarr_enc)
	
		with open(gt_fn) as gt :
			ground_truth = json.load(gt)

		for i in range(len(ground_truth['globules'])) :
			if (ground_truth['globules'][i] == 1) :
				if (g_prediction[i] >= 0.2) :
					g_truepos += 1
					g_imarr_overlay[imarr_dec == i] = [255, 255, 255]
				else :
					g_falseneg += 1
					g_imarr_overlay[imarr_dec == i] = [255, 0, 0]
			else :
				if (g_prediction[i] < 0.2) :
					g_trueneg += 1
					g_imarr_overlay[imarr_dec == i ] = [0,0,0]
				else :
					g_falsepos += 1
					g_imarr_overlay[imarr_dec == i] = [0, 255, 0]

			if (ground_truth['streaks'][i] == 1) :
				if (s_prediction[i] >= 0.2) :
					s_truepos += 1
					s_imarr_overlay[imarr_dec == i] = [255, 255, 255]
				else :
					s_falseneg += 1
					s_imarr_overlay[imarr_dec == i] = [255, 0, 0]
			else :
				if (s_prediction[i] < 0.2) :
					s_trueneg += 1 
					s_imarr_overlay[imarr_dec == i] = [0,0,0]
				else :
					s_falsepos += 1
					s_imarr_overlay[imarr_dec == i] = [0, 255, 0]


		gtp = float(g_truepos)
		gtn = float(g_trueneg)
		gfp = float(g_falsepos)
		gfn = float(g_falseneg)
		stp = float(s_truepos)
		stn = float(s_trueneg)
		sfp = float(s_falsepos)
		sfn = float(s_falseneg)

		g_accuracy = (gtp + gtn)/(gtp + gfp + gtn + gfn)
		if (gtp+gfn > 0) :
			g_sensitivity = gtp/(gtp + gfn)
		else :
			g_sensitivity = 0
		g_specificity = gtn/(gtn + gfp)
		#g_auc = roc_auc_score(ground_truth['globules'], g_prediction)
		#g_ap = average_precision_score(ground_truth['globules'], g_prediction)
		if (gtp + gfn + gfp > 0) :
			g_jaccard_ind = gtp/(gtp + gfn + gfp)
		else :
			g_jaccard_ind = 0


		s_accuracy = (stp + stn)/(stp + sfp + stn + sfn)
		if (stp + sfn > 0) :
			s_sensitivity = stp/(stp + sfn)
		else :
			s_sensitivity = 0
		s_specificity = stn/(stn + sfp)
		#s_auc = roc_auc_score(ground_truth['streaks'], s_prediction)
		#s_ap = average_precision_score(ground_truth['streaks'], s_prediction)
		if (stp + sfn + sfp > 0) :
			s_jaccard_ind = stp/(stp + sfn + sfp)
		else :
			s_jaccard_ind = 0

		print root_name, ',', 'globule', g_accuracy, ',', g_sensitivity, ',', g_specificity, ',', g_jaccard_ind

		print root_name, ',', 'streak', s_accuracy, ',', s_sensitivity, ',', s_specificity, ',', s_jaccard_ind



		g_image = Image.fromarray(g_imarr_overlay)
		s_image = Image.fromarray(s_imarr_overlay)

		#g_image.show()
		#s_image.show()

		g_image.save(root_name + '_globule_error.jpg')	
		s_image.save(root_name + '_streak_error.jpg')

