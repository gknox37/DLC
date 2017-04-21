#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include  "../../schemes/fixedPointCompress.h"

#define M_SIZE 5000
#define RAND_RANGE 300 
#define RAND_MIN 0
#define PRECISION 1

int main(){
	float x = (float) 0b00010000;
	std::cout.precision(PRECISION);
    //creating example input
    float* matrix =(float*) malloc(M_SIZE*sizeof(float));
    //uint32_t* decompressedMatrix =(uint32_t*) malloc(M_SIZE *sizeof(uint32_t));
    //uint8_t* compressedMatrix = (uint8_t*) malloc( M_SIZE *(sizeof(uint32_t) + 4 ));
    //unsigned* pointers = (unsigned*) malloc((M_SIZE) * sizeof(unsigned));
    
    for (int row = 0; row < M_SIZE; row++){
		float temp = rand() % RAND_RANGE + RAND_MIN;
    	matrix[row] = temp;
		std::cout << "Val: "<< std::fixed << (float)x << "\n";
            //printf("Val :%d\n", matrix[row*M_SIZE +col]);
    }
	return 0;

	/*
	clock_t begin = clock();
    unsigned numBytes = compressMatrix32_row(matrix, M_SIZE, N_SIZE, compressedMatrix, pointers );
    int retVal = decompressMatrix32_row(compressedMatrix, pointers, M_SIZE, N_SIZE, decompressedMatrix);
	clock_t end = clock();

    if(retVal == -1){
        printf("Couldn't decompress");
        return -1;
    }


    //comparing matrices
    for (unsigned row = 0; row< M_SIZE; row++){
        for(unsigned col = 0; col< N_SIZE; col++){
            if(matrix[row*M_SIZE+col] != decompressedMatrix[row*M_SIZE+col]){
                printf("Difference row:%d col:%d\n", row, col);
                printf("orig:%d new:%d\n",matrix[row*M_SIZE+col] , decompressedMatrix[row*M_SIZE+col]);
                free(matrix);
                free(compressedMatrix);
                free(pointers);
                return -1; 
            }
        }
    }

	double timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
    double ratio = ((double)(numBytes + M_SIZE*4))/ ((double)(M_SIZE*N_SIZE*4));

	printf("Time: %f ms\n", timeSpent);
    printf("Number of bytes in compressed matrix = %d\n", numBytes);
    printf("Number of bytes in pointer matrix = %d\n", M_SIZE * 4);
    printf("Number of bytes in original matrix = %d\n", M_SIZE*N_SIZE * 4);
    printf("Compression ratio = %f\n", ratio);
	*/
    free(matrix);
    //free(compressedMatrix);
	//free(decompressedMatrix);
    //free(pointers);
		
}
