import numpy as np
from skimage.segmentation import slic
from PIL import Image
from sys import platform as _platform
import argparse
from sklearn.metrics import roc_auc_score

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


for fn in file_list:
    if fn.endswith('.jpg'):

		root_name = fn.strip('.jpg')
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

		for i in range(len(ground_truth['globules'][i])) :
			if (ground_file['globules'][i] == 1) :
				if (g_prediction[i] >= 0.5) :
					g_truepos += 1
					g_imarr_overlay[imarr_dec == i] = [255, 255, 255]
				else :
					g_falseneg += 1
					g_imarr_overlay[imarr_dec == i] = [255, 0, 0]
			else :
				if (g_prediction[i] < 0.5)
					g_trueneg += 1
				else :
					g_falsepos += 1
					g_imarr_overlay[imarr_dec == i] = [0, 255, 0]

			if (ground_file['streaks'][i] == 1) :
				if (s_prediction[i] >= 0.5) :
					s_truepos += 1
					s_imarr_overlay[imarr_dec == i] = [255, 255, 255]
				else :
					s_falseneg += 1
					s_imarr_overlay[imarr_dec == i] = [255, 0, 0]
			else :
				if (s_prediction[i] < 0.5)
					s_trueneg += 1
				else :
					s_falsepos += 1
					s_imarr_overlay[imarr_dec == i] = [0, 255, 255]


		gtp = float(g_truepos)
		gtn = float(g_trueneg)
		gfp = float(g_falsepos)
		gfn = float(g_falseneg)
		stp = float(s_truepos)
		stn = float(s_trueneg)
		sfp = float(s_falsepos)
		sfn = float(s_falseneg)

		g_accuracy = (gtp + gtn)/(gtp + gfp + gtn + gfn)
		g_sensitivity = gtp/(gtp + gfn)
		g_specificity = gtn/(gtn + gfp)
		g_auc = roc_auc_score(ground_file['globules'], g_prediction)
		g_ap = average_precision_score(ground_file['globules'], g_prediction)

		s_accuracy = (stp + stn)/(stp + sfp + stn + sfn)
		s_sensitivity = stp/(stp + sfn)
		s_specificity = stn/(stn + sfp)
		s_auc = roc_auc_score(ground_file['streaks'], s_prediction)
		s_ap = average_precision_score(ground_file['streaks'], s_prediction)

		g_image = Image.fromarray(g_imarr_overlay)
		s_image = Image.fromarray(s_imarr_overlay)

		g_image.show()
		s_image.show()

		g_image.save(root_name + '_globule_error.jpg')	
		s_image.save(rootname + '_streak_error.jpg')

