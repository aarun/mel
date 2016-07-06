from sklearn.ensemble import RandomForestRegressor
import numpy as np
from PIL import Image
import argparse
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
image = img_as_float(io.imread(args["image"]))
segments = slic(image, n_segments = 3000, sigma = 5, slic_zero = 2)

xr = len(image)
#print xr
yr = len(image[0])


data = np.loadtxt("output.csv", delimiter= ",")
l = len(data)
data.resize(l, 4)

result = np.loadtxt("output1.csv", delimiter= ",")
lr = len(result)
result.resize(lr,)

est = RandomForestRegressor(n_estimators = 500)



est.fit(data, result)

print "Finished planting!"

pred = np.loadtxt("output5.csv", delimiter= ",")
lp = len(pred)
pred.resize(lp, 4)

predict = est.predict(pred)



#predict = predict.resize(767, 1022)

data2 = np.zeros( (xr,yr), dtype=np.uint8 )

#predict = predict.astype('uint8')*255 
counter = 0
prevlab = 0


for i in range(0, xr):
    for j in range(0, yr): 
    	lbl = segments[i][j]
    	#print(lbl)
    	val = predict[lbl]
    	#print(val)
    	if (val > 0.5):
    		data2[i][j] = 1
    		predict[lbl] = 1

    	else:
    		data2[i][j] = 0
    		predict[lbl] = 0


    	
    	#data2[i][j] = val



w = csv.writer(open("output2.csv", "w"))
for i in range(len(predict)):
	#if (labels[i, 0] == 1):
	#	w.writerow("1")
	#else:
	#	w.writerow("0")
	w.writerow([predict[i]])


data2 = data2.astype('uint8')*255 

im = Image.fromarray(data2)

im.show()



