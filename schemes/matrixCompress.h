#include "stdio.h"

#ifndef matrixCompress_H
#define matrixCompress_H

int test();

unsigned compressMatrix32_row(uint32_t* matrix, unsigned M,  unsigned N, uint8_t* compressedMatrix, unsigned* pointers);


#endif //matrixCompresss_H
