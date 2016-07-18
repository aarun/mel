import os


counter = 1
counter2 = 0
batch_str = ('')

f_out=open('batch1.txt', 'w')

for fn in os.listdir('.') :
	if (fn.endswith('.jpg') and fn[0] != '.') :
		if counter2 < 50 :
			batch_str += (str(fn) + '\n')
			counter2 +=1
		else :
			f_out.write(batch_str)
			f_out.close()
			counter2 = 0
			counter +=1
			f_out=open('batch' + str(counter) + '.txt', 'w')
			batch_str = ('')

f_out.write(batch_str)
f_out.close()