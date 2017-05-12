import sys
from random import randint
num_elements = int(sys.argv[1])
f = open('data/input_' + str(num_elements) +'_random'+'.raw','w')
f.write(str(num_elements)+'\a')
for i in range (0 , num_elements):
        if((i/1000)%2==0):
		f.write(str(100*(i%31)*randint(0,100000)) + '\n')
        else:
		f.write(str(randint(0,310000)*(i%31)) + '\n')
f.close()
