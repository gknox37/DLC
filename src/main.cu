#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <valarray>
#include <cuda_profiler_api.h>

#include "range.hpp"
#include "utils.hpp"

#define blockSize 64  // Power of 2 ?


void initArray(int size , uint8_t **ptrArray1 , uint16_t ** ptrArray2 , bool *isCompressed)
{
    for ( int i =0 ; i < size ; i++)
    {
        ptrArray1[i] = NULL;
        ptrArray2[i] = NULL;
        isCompressed[i] = false ;
    }
}


void bdCompress(char* c , int len , uint8_t ** ptrArray1 , uint16_t **ptrArray2 , bool* isCompressed , long* baseVals)
{
    char* temp;
    int blkCounter = 0 ;
    int numBlocks = len/blockSize ;
    int i ;
    int offset = 0 ;
    int numPtrs = blockSize/(sizeof(long)) ;
    long* ptrArray[numPtrs] ;
    
    
    while(offset + blockSize <= len) // Don't want to compress if length remaining is less than block
    {
        
        // Assume each value is 4 bytes , long , can try to vary this later.
        // Try to compress it to unsigned int_8 (0-255)
        // Get the minimum value as base so unsigned int 8 can be used for deltas
        // If value ranges are more than max value of unsigned int 8 , try to convert to unsigned int.
        // If ranges are more than unsigned int , don't compress this block , move onto next block.
        
        for ( i = 0 ; i < numPtrs ; i++)
        {
            ptrArray[i] = (long*) (&c[offset + i*sizeof(long)]) ;
        }
        bool flag = false ;
        long minValue ;
        for ( i =0 ; i<numPtrs ; i++)
        {
            if(flag==false)
            {
                flag = true ;
                minValue = *(ptrArray[i]);
            }
            else
            {
                if(*(ptrArray[i]) < minValue)
                minValue = (*(ptrArray[i])) ;
            }
        }
        
        long range = 0 ;
        flag = false ;
        for (i =0 ; i<numPtrs ; i++)
        {
            if(flag ==false)
            {
                range = (*(ptrArray[i])) - minValue ;
                flag = true ;
            }
            else
            {
                if((*(ptrArray[i])) - minValue > range)
                range = *(ptrArray[i]) - minValue ;
            }
        }
        if(range < pow(2 , sizeof(uint8_t) * 8))
        // compress into uint8
        {
            isCompressed[blkCounter] = true ;
            baseVals[blkCounter] = minValue ;
            ptrArray1[blkCounter] = (uint8_t*) malloc ( sizeof(uint8_t)*numPtrs) ;
            for ( i = 0 ; i < numPtrs ; i++)
            (ptrArray1[blkCounter])[i] = (*(ptrArray[i])) - minValue ;
            
        }
        else if (range < pow(2,sizeof(uint16_t)*8))
        {
            // compress into uint16
            isCompressed[blkCounter] = true ;
            baseVals[blkCounter] = minValue ;
            ptrArray2[blkCounter] = (uint16_t*)malloc ( sizeof(uint16_t)*numPtrs) ;
            for ( i = 0 ; i < numPtrs ; i++)
            (ptrArray2[blkCounter])[i] = (*(ptrArray[i])) - minValue ;
        }
        
        offset += blockSize ;
        blkCounter ++ ;
    }
}


int main(int argc, char **argv) {
    
    // get start time
    const auto start = now();
    
    int longArraySize = 16 ;
    long testArray[longArraySize] ;
    int i ,j ;
    for ( i =0 ; i < longArraySize ; i++)
    testArray[i] = 5*i ;
    int numBlocks = (longArraySize * sizeof(long))/blockSize ;
    uint8_t * ptrArray1[numBlocks] ;
    uint16_t * ptrArray2[numBlocks] ;
    long baseVals[numBlocks] ;
    bool isCompressed[numBlocks] ;
    initArray(numBlocks , ptrArray1 , ptrArray2 , isCompressed) ;
    
    bdCompress((char*)testArray , longArraySize * sizeof(long) , ptrArray1 , ptrArray2 , isCompressed , baseVals) ;
    printf( "Numblocks :%d\n" , numBlocks);
    for ( i = 0; i<numBlocks ; i++)
    {
        printf("Bases : %d ," , baseVals[i]);
    }
    printf("\n") ;
    int numPtrs = blockSize/sizeof(long) ;
    for ( i =0 ; i < numBlocks ; i++)
    {
        for (  j =0 ; j < numPtrs ; j++)
        printf("Deltas : %d , ", (int)((ptrArray1[i])[j])) ;
    }
    int bytesAfterCompression = numBlocks * ( numPtrs + sizeof(uint8_t*)  + sizeof(bool) + sizeof(long)) ;
    int bytesBeforeCompression =  longArraySize * sizeof(long) ;
    double ratio = ((double)bytesAfterCompression)/((double)bytesBeforeCompression) ;
    printf("Ratio : %f\n" , ratio);
    
    printf("\n");
    
    // get end time
    const auto end = now();
    // get elapsed time in milliseconds
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Compression time elapsed = " << elapsed << " milliseconds.";
    return 0;
}
