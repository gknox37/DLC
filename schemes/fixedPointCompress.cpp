#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include "fixedPointCompress.h"

// assuming fixed point 32-bit numbers
uint32_t compressFixed32(uint32_t* in, uint32_t len, uint8_t precision, uint8_t loss, uint32_t batchSize, uint32_t* out, uint32_t* pointers){	
	uint32_t numBytes = 0;
	uint32_t pointersIdx = 0;
	std::cout.precision(precision);
    

	//compressing batchSize by batchSize
	for(uint32_t inIdx = 0; inIdx < len; inIdx += batchSize){
		pointers[pointersIdx]  = numBytes; 		
		
		uint32_t min = in[inIdx];
		uint32_t max = in[inIdx];

		//calculate base
		for(uint32_t currInIdx = 1 + inIdx ; currInIdx < batchSize + inIdx; currInIdx++){
			if( in[currInIdx] > max)
				max = in[currInIdx];

			if( in[currInIdx] > max)
				min = in[currInIdx];
		}
		
		std::cout<< "Min:  " << std::fixed << min << " Max: " << std::fixed << max << " Precision: " << precision << "\n";
		

		//setup
		pointersIdx++;
	}
	return numBytes;
}
