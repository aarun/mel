import numpy as np

stnd = np.loadtxt("output1.csv", delimiter= ",")

chck = np.loadtxt("output2.csv", delimiter= ",")

val = 0

for i in range(len(stnd)):
	if (stnd[i] != chck[i]):
		val += 1

print(val)

