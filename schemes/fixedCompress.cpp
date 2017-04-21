#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include "fixedCompress.h"

// assuming fixed point 32-bit numbers
uint32_t compressFixed32(uint32_t* in, uint32_t len, uint8_t decimalPos, uint8_t precisionLoss, uint32_t batchSize, uint32_t* out, uint32_t* pointers){	
	uint32_t numBytes = 0;
	uint32_t pointersIdx = 0;

	//compressing batchSize by batchSize
	for(int inIdx = 0; inIdx < size; inIdx += batchSize){
		pointers[pointersIdx] 		
		
		uint32_t min = in[inIdx];
		uint32_t max = in[inIdx];

		//calculate base
		for(int currInIdx = 1 + inIdx ; currInIdx < batchSize + inIdx; currInIdx++){
			if( in[currInIdx] > max)
				max = in[currInIdx];

			if( in[currInIdx] > max)
				min = in[currInIdx];
		}

		print()

		//setup
		pointersIdx++;
	}
	return numBytes;
}
