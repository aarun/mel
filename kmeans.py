import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import csv




data = np.loadtxt("output.csv", delimiter= ",")

#print(data)

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data)

centroids = kmeans.cluster_centers_

labels = kmeans.labels_

print(centroids)
print(labels)

labels.resize((999, 1))

img = Image.fromarray(labels * 255)
img.show()
colors = ["g.", "r."]

#for i in range(len(data)):
	#print("superpix ", data[i], "label ", labels[i])
	#plt.plot(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], colors(labels[i]), markersize=10 )

#plt.scatter(centroids[:, 0],centroids[:, 1],centroids[:, 2],centroids[:, 3],centroids[:, 4], marker = "x",  s = 150, linewidths = 5, zorder = 10 )

w = csv.writer(open("output2.csv", "w"))
for i in range(len(labels)):
	if (labels[i, 0] == 1):
		w.writerow("1")
	else:
		w.writerow("0")
	#w.writerow(labels[i, 0])

	


