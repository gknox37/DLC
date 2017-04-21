#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include  "../schemes/matrixCompress.h"

#define M_SIZE 500
#define N_SIZE 500
#define RAND_RANGE 300 
#define RAND_MIN 0


int main(){
    //creating example input
    uint32_t* matrix =(uint32_t*) malloc(M_SIZE*N_SIZE *sizeof(uint32_t));
    uint32_t* decompressedMatrix =(uint32_t*) malloc(M_SIZE*N_SIZE *sizeof(uint32_t));
    uint8_t* compressedMatrix = (uint8_t*) malloc( M_SIZE*5 + M_SIZE* N_SIZE * sizeof(uint32_t));
    unsigned* pointers = (unsigned*) malloc((M_SIZE) * sizeof(unsigned));
    
    for (int row = 0; row < M_SIZE; row++){
        for (int col = 0; col< N_SIZE; col++){
            matrix[row*M_SIZE + col] = rand() % RAND_RANGE + RAND_MIN;
            //printf("Val :%d\n", matrix[row*M_SIZE +col]);
        }
    }

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

    free(matrix);
    free(compressedMatrix);
    free(pointers);
		
}
