import numpy as np

stnd = np.loadtxt("output1.csv", delimiter= ",")

chck = np.loadtxt("output2.csv", delimiter= ",")

falsepos = 0
falseneg = 0

for i in range(len(stnd)):
	if (stnd[i] == 1):
		if (chck[i] != 0):
			falseneg += 1 
	if (stnd[i] == 0):
		#print "do"
		if (chck[i] == 0):
			print "hello"
			falsepos+= 1



print(falsepos)
print(falseneg)

