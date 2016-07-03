import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import csv
import pylab as pl
import pandas
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import argparse
from skimage.util import img_as_float
from skimage import io

#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#args = vars(ap.parse_args())

#image = img_as_float(io.imread(args["image"]))

v = (np.random.random((4,100))-0.5)*15

data = np.loadtxt("output.csv", delimiter= ",")
l = len(data)
data.resize(l, 4)
pca = PCA(n_components=2).fit(data)
pca_2d = pca.transform(data)
#print(len(data))
#print(len(data[0]))

kmeans = KMeans(n_clusters = 4)

kmeans.fit(data)



centroids = kmeans.cluster_centers_
print(centroids)

#for i in range(len(centroids)):


labels = kmeans.labels_

pl.figure('Reference Plot')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1],c=labels)
pl.show()

#for i in range(0, pca_2d.shape[0]):
#	if labels[i] == 0:
#		c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
#	elif labels[i] == 1:
#		c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
#	elif labels[i] == 2:
#		c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='*')
#	pl.legend([c1, c2, c3], ['Setosa', 'Versicolor','Virginica'])
#	pl.title('Iris dataset with 3 clusters and known outcomes')
#	pl.show()









#print(centroids)
#print(labels)
#N = 50
#x = randn(N)
#y = randn(N)
#z = randn(N)
#colors = [x, y, z]

labels.resize((l, 1))

img = Image.fromarray(labels * 255)
#img.show()
#colors = np.random.rand(N)

#c = np.abs(v)

k_means_labels = kmeans.labels_
k_means_cluster_centers = kmeans.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)








#plt.subplot(221)
#plt.scatter(data[:, 0], data[:, 1], data[:, 2],  c=colors, cmap= kmeans)
#plt.show()
#fig = plt.figure(figsize=(8, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#fig = plt.figure()
#ax = fig.add_subplot(1, 3, 1)
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)

#fig.show()

#Axes3D.scatter(data[:, 0], data[:, 1], zs=data[:, 2], zdir='z', s=20, c=colors, depthshade=True)

#for k, col in zip(range(len(centroids)), colors):
#
  #  my_members = k_means_labels == k
  #  print(my_members)
  #  cluster_center = k_means_cluster_centers[k]
  #  ax.plot(data[my_members, 0], data[my_members, 1], 'w',
  #          markerfacecolor=col, marker='.')
  #  ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
  #          markeredgecolor='k', markersize=6)
#ax.set_title('KMeans')
#ax.set_xticks(())
#ax.set_yticks(())
#plt.show()

#for i in range(len(data)):
	#print("superpix ", data[i], "label ", labels[i])
	#plt.plot(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], colors(labels[i]), markersize=10 )

#plt.scatter(centroids[:, 0],centroids[:, 1],centroids[:, 2],centroids[:, 3],centroids[:, 4], marker = "x",  s = 150, linewidths = 5, zorder = 10 )

w = csv.writer(open("output2.csv", "w"))
for i in range(len(labels)):
	#if (labels[i, 0] == 1):
	#	w.writerow("1")
	#else:
	#	w.writerow("0")
	w.writerow([labels[i, 0]])

	


