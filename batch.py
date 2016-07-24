import os


counter = 1
counter2 = 0
batch_str = ('')

f_out=open('batch1.txt', 'w')

for fn in os.listdir('.') :
	if (fn.endswith('.jpg') and fn[0] != '.') :
		if (fn.endswith('r.jpg')) :
			batch_str += ''
		else :
			if counter2 < 500 :
				batch_str += (str(fn) + '\n')
				counter2 +=1
			else :
				batch_str += (str(fn) + '\n')
				f_out.write(batch_str)
				f_out.close()
				counter2 = 0
				counter +=1
				f_out=open('batch' + str(counter) + '.txt', 'w')
				batch_str = ('')

f_out.write(batch_str)
f_out.close()