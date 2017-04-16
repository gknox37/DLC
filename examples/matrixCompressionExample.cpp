#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include  "../schemes/matrixCompress.h"

#define M_SIZE 10
#define N_SIZE 10
#define RAND_RANGE 100 
#define RAND_MIN 0


int main(){
    //creating example input
    uint32_t* matrix =(uint32_t*) malloc(M_SIZE*N_SIZE *sizeof(uint32_t));
    uint8_t* compressedMatrix = (uint8_t*) malloc((M_SIZE + 1)* N_SIZE * sizeof(uint8_t));
    unsigned* pointers = (unsigned*) malloc((M_SIZE) * sizeof(unsigned));
    
    for (int row = 0; row < M_SIZE; row++){
        for (int col = 0; col< N_SIZE; col++){
            matrix[row*M_SIZE + col] = rand() % RAND_RANGE + RAND_MIN;
            printf("Val :%d\n", matrix[row*M_SIZE +col]);
        }
    }

	clock_t begin = clock();
    unsigned numBytes = compressMatrix32_row(matrix, M_SIZE, N_SIZE, compressedMatrix, pointers );
	clock_t end = clock();
	double timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
	printf("Time: %f ms\n", timeSpent);

    free(matrix);
    free(compressedMatrix);
    free(pointers);
		
}
