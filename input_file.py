import sys
num_elements = int(sys.argv[1])
f = open('data/input_' + str(num_elements) +'.raw','w')
f.write(str(num_elements)+'\n')
for i in range (0 , num_elements):
        if((i/1000)%2==0):
		f.write(str(100*(i%31)) + '\n')
        else:
		f.write(str(10*(i%31)) + '\n')
f.close()
