#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include  "fixedPointCompress.h"

#define M_SIZE 10
#define RANDO_MIN  -500
#define RANDO_MAX  500
#define BATCH_SIZE 10

int main(){
    //creating example input
    srand( time(NULL) );
    fixed_point24_8* in = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
    fixed_point24_8* in_decom = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
    uint8_t* compressed_in = (uint8_t*) malloc(sizeof(fixed_point24_8)*M_SIZE*2 );//2 for safety
    unsigned* pointers = (unsigned*) malloc((M_SIZE) * sizeof(unsigned));
    
    for (int idx  = 0; idx < M_SIZE; idx++){
        float currRand =  RANDO_MIN + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(RANDO_MAX-RANDO_MIN)));
        in[idx].insert(currRand); 
        printf("Val :%f\n", currRand);
    }

	clock_t begin = clock();
    int numBytes = compressFixed24_8(in,M_SIZE,0,BATCH_SIZE,compressed_in,pointers);
    printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
    int retVal = decompressFixed24_8(compressed_in, pointers, M_SIZE, in_decom, BATCH_SIZE);
	clock_t end = clock();

    if(retVal == -1){
        printf("Couldn't decompress");
        return -1;
    }

    //comparing matricies
    for(int i = 0; i < M_SIZE; i++){
        int sub = in[i].data - in_decom[i].data; 
        printf("i=%d| %d - %d = %d\n", i, in[i].data, in_decom[i].data, sub);
        if(sub != 0){
            printf("ERROROROROROR\n");
            return -1;
        }
    }
    printf("Matricies match!!!\n");
    return 0;
    /*

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
    free(in);
    //free(compressedMatrix);
	//free(decompressedMatrix);
    //free(pointers);
		
}
