#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include "fixedPointCompress.h"

// assuming fixed point 32-bit numbers
uint32_t compressFixed24_8(fixed_point24_8* in, uint32_t len, uint8_t loss, uint32_t batchSize, uint8_t* out, unsigned* pointers){	
	uint32_t numBytes = 0;
	uint32_t pointersIdx = 0;
    char currByte;
    char compressable = 1;
    int curr,diff; 
    int currBase;
    
	//compressing elemenets of in into batches of size batchsize
	for(uint32_t inIdx = 0; inIdx < len; inIdx += batchSize){
        pointers[pointersIdx] = numBytes;

        //assuming best case compression
        compressable = 1; 

        //looping through in to check compression
        for(int batchIdx = inIdx; batchIdx < inIdx + batchSize && batchIdx < len; batchIdx++){
            curr = in[batchIdx].data;

            //Checking for positive and negative bases
            diff = abs((curr >> 8) - base);
            //printf("Diff is %d\n",diff);
            //printf("Float value is %f\n", in[batchIdx].convert());
            //printf("Fractional protion %x\n",curr & ~mask24_8);
            if(diff < pow(2,7) -1 ){
            }
            else if(diff < pow(2,15) -1 ){
                compressable = 2;
            }
            else{
                compressable = 3;
                break;
            }
        }

        printf("Batch is %d byte compressable\n",compressable);

        //writing batch info bit 
        memcpy(&out[numBytes], &compressable, 1);
        numBytes++;

        //compressing data
        for(int batchIdx = inIdx; batchIdx < inIdx + batchSize && batchIdx < len; batchIdx++){
            curr = in[batchIdx].data;
            
            diff = (curr >> 8) - base; 
            
            //pointing to valid data
            uint8_t* currBytePointer = (uint8_t*)&diff;
            currBytePointer += (compressable-1); //little endian

            //writing the bytes
            memcpy(&out[numBytes],currBytePointer,compressable);
            numBytes += compressable;

            //writing the fractional component
            currByte = curr & ~mask24_8;
            memcpy(&out[numBytes], &currByte, 1);
            numBytes++;

            printf("-----------------------\n");
            printf("Delta Pre compress is %d\n",diff);
            printf("Frac Pre compress is %x\n",currByte);
            //printf("Value written  = %x\n", *currBytePointer);
            printf("Data Pre compress is = %x\n",curr);
        }
    }
	return numBytes;
}

uint32_t decompressFixed24_8(uint8_t* in, unsigned* pointers, unsigned len, fixed_point24_8* out, uint32_t batchSize){
    int baseIdx = 0;
    int8_t currByte, currFrac;
    int16_t currShort;
    int currData;
    int numBatches = ceil(len/batchSize);

    for (int currBatch = 0; currBatch < numBatches; currBatch++){
        //finding encoding
        baseIdx = pointers[currBatch];
        char compressable = in[baseIdx];
        baseIdx++; 

        printf("Batch compression is %d\n",compressable);
        for(int currElem = currBatch*batchSize; currElem < currBatch*batchSize + batchSize && currElem < len; currElem++){
            if(compressable == 1){
                memcpy(&currByte,&in[baseIdx],1);
                baseIdx += 1;
                memcpy(&currFrac,&in[baseIdx],1);
                baseIdx += 1;
                out[currElem].data = (((int)currByte +base) << 8) | (~mask24_8 & (int)currFrac); 
                printf("-----------------------\n");
                printf("Delta after compression = %d\n",currByte);
                printf("Frac after compression = %x\n",currFrac);
                printf("Data after compression = %x\n",out[currElem].data);
             }
            else if(compressable == 2){
                memcpy(&currShort,&in[baseIdx],2);
                baseIdx += 2;
                memcpy(&currFrac,&in[baseIdx],2);
                baseIdx += 2;
                out[currElem].data = (((int)currShort +base) << 8) | (~mask24_8 & (int)currFrac); 
            }
            else if(compressable == 3){
                memcpy(&out[currElem].data,&in[baseIdx],4);
                baseIdx += 4;
            }
            else{
                printf("Unrecongized Compression\n");
                return -1;
            }
        }
    }
    printf("Compression Complete\n");
    return 0; 
}
