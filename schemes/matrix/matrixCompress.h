#include "stdio.h"

#ifndef matrixCompress_H
#define matrixCompress_H


#define COMP32 0
#define COMP16 1
#define COMP8 2

unsigned compressMatrix32_row(uint32_t* matrix, unsigned M,  unsigned N, uint8_t* compressedMatrix, unsigned* pointers);


int decompressMatrix32_row(uint8_t* compressedMatrix, unsigned* pointers, unsigned M, unsigned N, uint32_t* matrix);
#endif //matrixCompresss_H
