import numpy as np
from skimage.segmentation import slic
from PIL import Image


#stnd = np.loadtxt("output6.csv", delimiter= ",")

prediction = np.loadtxt("ISIC_0000089_Prediction.csv", delimiter= ",")

falsepos = 0
falseneg = 0
trueneg = 0
truepos = 0

gt = Image.open('ISIC_0000089_Segmentation.png')
original = Image.open('/Users/sahana/Mel/ISBI2016_ISIC_Part1_Training_Data/ISIC_0000089.jpg')
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

print truepos
print falseneg
print trueneg
print falsepos

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
image.show()

#for i in range(len(stnd)):
#	if (stnd[i] == 1):
#		if (chck[i] != 1):
#			falseneg += 1 
#	if (stnd[i] == 1):
#		#print "do"
#		if (chck[i] == 0):
#			
#			falsepos+= 1

print accuracy
print sensitivity
print specificity
print dice_coeff
print jaccard_ind




