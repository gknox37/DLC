#------------------------------------------------------------------------------

CC=g++
CFLAGS =-g -Wall
MYPROGRAM=fixedPointCompressionExample
MYINCLUDES=-I ../../schemes/fixedPoint/
SOURCE = fixedPointCompressionExample.cpp  
REQ = fixedPointCompress.h
MYLIBRARIES=lftk  


#------------------------------------------------------------------------------



all: $(MYPROGRAM)

fixedPointCompressionExample: fixedPointCompressionExample.o fixedPointCompress.o
	$(CC) $(CFLAGS) $(MYINCLUDES) -o fixedPointCompressionExample fixedPointCompressionExample.o fixedPointCompress.o

fixedPointCompressionExample.o: fixedPointCompressionExample.cpp
	$(CC) $(CFLAGS) $(MYINCLUDES) -c fixedPointCompressionExample.cpp

fixedPointCompress.o: ../../schemes/fixedPoint/fixedPointCompress.cpp  ../../schemes/fixedPoint/fixedPointCompress.h
	$(CC) $(CFLAGS) $(MYINCLUDES) -c ../../schemes/fixedPoint/fixedPointCompress.cpp
	
clean:
	$(RM) *.o fixedPointCompressionExample
	
	
	
