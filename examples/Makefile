#------------------------------------------------------------------------------

CC=g++
CFLAGS =-g -Wall
MYPROGRAM=matrixCompressionExample
MYINCLUDES=-I./../schemes
SOURCE = matrixCompressionExample.cpp  
REQ = matrixCompress.h
MYLIBRARIES=lftk  


#------------------------------------------------------------------------------



all: $(MYPROGRAM)


matrixCompressionExample: matrixCompressionExample.o matrixCompress.o
	$(CC) $(CFLAGS) $(MYINCLUDES) -o matrixCompressionExample matrixCompressionExample.o matrixCompress.o

matrixCompressionExample.o: matrixCompressionExample.cpp
	$(CC) $(CFLAGS) $(MYINCLUDES) -c matrixCompressionExample.cpp

matrixCompress.o: ../schemes/matrixCompress.cpp ../schemes/matrixCompress.h
	$(CC) $(CFLAGS) $(MYINCLUDES) -c ../schemes/matrixCompress.cpp
	
clean:
	$(RM) *.o matrixCompressionExample
	
	
	
