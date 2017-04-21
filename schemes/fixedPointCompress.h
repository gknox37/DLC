#include <stdio.h>

#ifndef FIXED_POINT_COMPRESS_H
#define FIXED_POINT_COMPRESS_H


uint32_t compressFixed32(uint32_t* in, uint32_t len, uint8_t precision, uint8_t loss, uint32_t batchSize, uint32_t* out, uint32_t* pointers);



#endif // FIXED_POINT_COMPRESS_
