#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include "matrixCompress.h"





unsigned compressMatrix32_row(uint32_t* matrix, unsigned M,  unsigned N, uint8_t* compressedMatrix, unsigned* pointers ){

    //Compressing row by row
    for(int row = 0; row < M; row++){

        //finding possible encoding
        uint32_t max = matrix[row*M + 0];
        uint32_t min = matrix[row*M + 0];
        for(int col = 1; col < N; col++){
            if(matrix[row*M + col] > max)
                max = matrix[row*M + col];

            if(matrix[row*M + col] < min)
                min = matrix[row*M + col];

            printf("Distance between max and min is %d\n", max-min);
        }

    }
    return 0;
}

int test(){
    return 32;
}
