#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include "matrixCompress.h"




unsigned compressMatrix32_row(uint32_t* matrix, unsigned M,  unsigned N, uint8_t* compressedMatrix, unsigned* pointers ){

    unsigned numBytes = 0;
    uint8_t tempByte = 0;
    uint16_t tempByte2 = 0;
    uint32_t tempByte4 = 0;

    //Compressing row by row
    for(unsigned row = 0; row < M; row++){
        //updating pointers
        pointers[row] = numBytes;
        //printf("index = %d\n", pointers[row]);

        //finding possible encoding
        uint32_t max = matrix[row*M + 0];
        uint32_t min = matrix[row*M + 0];
        for(unsigned col = 1; col < N; col++){
            if(matrix[row*M + col] > max)
                max = matrix[row*M + col];

            if(matrix[row*M + col] < min)
                min = matrix[row*M + col];
        }

        printf("Max - min: %d, min = :%d, row:%d\n", max-min, min, row);
        uint32_t base = min;

        // fourth size
        if(max-min < pow(2,8) -1){
            printf("row is byte compressable\n");
            //writing the encoding byte
            tempByte = COMP8;
            memcpy(&compressedMatrix[numBytes], &tempByte, 1);
            numBytes++;

            //writing the base
            memcpy(&compressedMatrix[numBytes], &base, 4);
            numBytes = numBytes + 4;
            //printf("row: %d, base: %d\n", row, base);

            //writing compressed numbers
            for(unsigned col = 0; col < N; col++){
                tempByte = matrix[row*M + col] - base; 
                memcpy(&compressedMatrix[numBytes], &tempByte, 1);
                numBytes++;
                //printf("Base: %d, Curr:  %d, delta: %d\n", base, matrix[row*M + col],tempByte);
            }
                
        }
        // half size
        else if (max-min < pow(2,16) -1){
            printf("row is 2 byte compressable\n");
            //writing the encoding byte
            tempByte = COMP16;
            memcpy(&compressedMatrix[numBytes], &tempByte, 1);
            numBytes++;

            //writing the base
            memcpy(&compressedMatrix[numBytes], &base, 4);
            numBytes = numBytes + 4;
            //printf("row: %d, base: %d\n", row, base);

            //writing compressed numbers
            for(unsigned col = 0; col < N; col++){
                tempByte2 = matrix[row*M + col] - base; 
                memcpy(&compressedMatrix[numBytes], &tempByte2, 2);
                numBytes = numBytes + 2;
                //printf("Base: %d, Curr:  %d, delta: %d\n", base, matrix[row*M + col],tempByte);
            }
                
        }

        //original encoding
        else {
            printf("row is not byte compressable\n");
            //writing the encoding byte
            tempByte = COMP32;
            memcpy(&compressedMatrix[numBytes], &tempByte, 1);
            numBytes++;

            //writing the base
            memcpy(&compressedMatrix[numBytes], &base, 4);
            numBytes = numBytes + 4;
            //printf("row: %d, base: %d\n", row, base);

            //writing compressed numbers
            for(unsigned col = 0; col < N; col++){
                tempByte4 = matrix[row*M + col] - base; 
                memcpy(&compressedMatrix[numBytes], &tempByte4, 4);
                numBytes = numBytes + 4;
                //printf("Base: %d, Curr:  %d, delta: %d\n", base, matrix[row*M + col],tempByte);
            }
                
        }

    }
    return numBytes;
}


int decompressMatrix32_row(uint8_t* compressedMatrix, unsigned* pointers, unsigned M, unsigned N, uint32_t* matrix){
    uint8_t mode = 0;
    unsigned baseIdx = 0;
    uint32_t base = 0;
    uint32_t delta = 0;

    for(unsigned row = 0; row < M; row++){
        baseIdx = pointers[row];
        mode = compressedMatrix[baseIdx];
        uint8_t byteSize = 0;

        if(mode == COMP8){
            byteSize = 1;
        }
        else if(mode == COMP16){
            byteSize = 2;
        }
        else if(mode == COMP32){
            byteSize = 4;
        }
        else{
            printf("Bad encoding");
            return -1;
        }

        // extrating info
        memcpy(&base,&compressedMatrix[baseIdx + 1], 4);
        //printf("baseIdx = %d\n", baseIdx);
        //printf("row: %d, base: %d\n", row, base);

        for(unsigned col = 0; col < N; col++){
            memcpy(&delta, &compressedMatrix[baseIdx + 1 + 4 + col*byteSize], byteSize);
            matrix[row*M + col] = delta + base;
        }

    }
    return 0;
}

