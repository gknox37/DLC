#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <valarray>
#include <fstream>
#include <cuda_profiler_api.h>

#include "range.hpp"
#include "utils.hpp"
using namespace std;

#define Algo_blockSize 512 //16 //This is number of bytes per compression block
#define SCAN_BLOCK_SIZE 512 //8
//#define blockSize 1024
//This is not number of threads in a ThreadBlock


void initArray(int size , int16_t *isCompressed)
{
   int i ;
   for ( i =0 ; i < size ; i++)
   {
     isCompressed[i] = 0 ;
   }
 }

//Casts t as int16, int32 or long, then deferences t.
long getVals(char * t , int base_size)
{
    long a ;
    if(base_size == 16)
        a = *((int16_t *) t) ; //deference after casting t as int16 pointer
    else if(base_size == 32)
        a = *((int32_t *) t) ; //deference after casting t as int32 pointer
    else
        a = *((long*)t) ;
    return a ;
}

//Simple scan kernel from 408
__global__ void scan_n(int16_t* input, float* output, int len){
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    __shared__ float partialSum[2*SCAN_BLOCK_SIZE]; //1024
    int tx = threadIdx.x;
    int start = 2*blockIdx.x*blockDim.x;
    int val;
    float CompressedBlockSize;

    if(blockIdx.x == 0 && tx == 0)
        partialSum[tx] = 0;
    if(start + tx >= len)
        partialSum[tx] = 0;
    else{
        val = (int) input[start+tx]; //[start+tx-1];
        if(val== 5 )
            //numElementsperBlock = Algo_blockSize/sizeof(uint16_t)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(uint16_t);
        else if(val== 9 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(uint32_t);
        else if(val== 13)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(long);
        else if(val== 6 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(uint16_t);
        else if(val== 10 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(uint32_t);
        else if(val== 14 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(long);
        else if(val== 7)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(uint16_t);
        else if(val== 11)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(uint32_t);
        else if(val== 15)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(long);
        else if(val<0)
            CompressedBlockSize = -val; //This is number of bytes
        partialSum[tx] = CompressedBlockSize;
    }
    if (start + blockDim.x + tx >= len)
        partialSum[tx+blockDim.x] = 0;
    else{
        val = (int) input[start+blockDim.x+tx]; //[start+blockDim.x+tx-1]
        if(val == 5 )
            //numElementsperBlock = Algo_blockSize/sizeof(uint16_t)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(uint16_t);
        else if(val == 9 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(uint32_t);
        else if(val == 13)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint8_t))/sizeof(long);
        else if(val== 6 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(uint16_t);
        else if(val== 10 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(uint32_t);
        else if(val== 14 )
            CompressedBlockSize = (Algo_blockSize*sizeof(uint16_t))/sizeof(long);
        else if(val== 7)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(uint16_t);
        else if(val== 11)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(uint32_t);
        else if(val== 15)
            CompressedBlockSize = (Algo_blockSize*sizeof(uint32_t))/sizeof(long);
        else if(val<0)
            CompressedBlockSize = -val; //This is number of bytes
        partialSum[tx+blockDim.x] = CompressedBlockSize;
/*        if( val == 5 or val == 9 or val == 13)
            partialSum[tx+blockDim.x] = (int) sizeof(uint8_t);
        else if( val==6 or val==10 or val==14)
            partialSum[tx+blockDim.x] = (int) sizeof(uint16_t);
        else if( val==7 or val==11 or val==15)
            partialSum[tx+blockDim.x] = (int) sizeof(uint32_t);
        else if (val <0)
            partialSum[tx+blockDim.x] = (int) -val;*/
    }
    __syncthreads();
    //PreScan setup
    int stride = 1;
    while (stride <= SCAN_BLOCK_SIZE){
        int index = (threadIdx.x+1) * stride * 2 - 1;
        if(index < SCAN_BLOCK_SIZE*2)
            partialSum[index] +=partialSum[index-stride];
        stride = stride*2;
        __syncthreads();
    }

    //PostScan setup
    stride = SCAN_BLOCK_SIZE/2;
    while(stride > 0){
        int index= (threadIdx.x+1)*stride*2-1;
        if (index+stride < SCAN_BLOCK_SIZE*2)
            partialSum[index+stride] += partialSum[index];
        stride = stride/2;
        __syncthreads();
    }
    if((tx+start) < len)
        output[tx+start] = partialSum[tx];
    if((start+SCAN_BLOCK_SIZE+tx) < len)
        output[start+SCAN_BLOCK_SIZE+tx] = partialSum[tx + SCAN_BLOCK_SIZE];
}


__global__ void scan_regular(float *input, float *output, int len) {

  __shared__ float partialSum[2*SCAN_BLOCK_SIZE]; //1024
  int tx = threadIdx.x;
  int start = 2*blockIdx.x*blockDim.x;

  if(start + tx >= len)
    partialSum[tx] = 0;
  else
    partialSum[tx] = input[start+tx];
  if (start + blockDim.x + tx >= len)
    partialSum[tx+blockDim.x] = 0;
  else
    partialSum[tx+blockDim.x] = input[start+blockDim.x+tx];

  __syncthreads();
  //PreScan setup
  int stride = 1;
  while (stride <= SCAN_BLOCK_SIZE){
    int index = (threadIdx.x+1) * stride * 2 - 1;
    if(index < SCAN_BLOCK_SIZE*2)
      partialSum[index] +=partialSum[index-stride];
    stride = stride*2;

    __syncthreads();
  }

  //PostScan setup
  stride = SCAN_BLOCK_SIZE/2;
  while(stride > 0){
    int index= (threadIdx.x+1)*stride*2-1;
    if (index+stride < SCAN_BLOCK_SIZE*2)
      partialSum[index+stride] += partialSum[index];
    stride = stride/2;
  __syncthreads();
  }
  if((tx+start) < len)
    output[tx+start] = partialSum[tx];
  if((start+SCAN_BLOCK_SIZE+tx) < len)
    output[start+SCAN_BLOCK_SIZE+tx] = partialSum[tx + SCAN_BLOCK_SIZE];
}

__global__ void ExtractLast(float *input, float *output, int len){
  //Want to get the last element out of Block.
  unsigned int index = (threadIdx.x + 1)*SCAN_BLOCK_SIZE*2 - 1;
  if (index < len) {
    output[threadIdx.x] = input[index];
  }
}

// Block size is 2 * BLOCK_SIZE.
__global__ void Finaladd(float *input, float *output, int len){
  if(blockIdx.x< ((len-1)/2*blockDim.x)) {
    output[threadIdx.x+blockIdx.x*blockDim.x+blockDim.x]+= input[blockIdx.x];
  }
}

/*__device__ int  scan_krishna ( int16_t * isCompressed , int numBlocks)
{
   // BytesSoFar is the array to perform the scan on
   int idx = blockIdx.x  ;
   int tx = threadIdx.x ;
   int localVal = 0 ;

   if ( idx < numBlocks && tx ==0)
   {
     if(isCompressed[idx] > 0) {
      localVal = isCompressed[idx] ; // get the isCompressed value
      int localChunkSize = localVal/4 ;
      if(localChunkSize == 1)
           localChunkSize = 2 ;
      else if (localChunkSize == 2)
           localChunkSize = 4;
      else
           localChunkSize = 8 ;
      int localCompSize = localVal % 4 ;
     if(localCompSize == 1)
        localCompSize = 1 ;
     else if (localCompSize == 2)
        localCompSize = 2 ;
    else
       localCompSize = 4 ;
    isCompressed[idx] = (blockSize/localChunkSize) * localCompSize ;
   }
  else if (isCompressed[idx]  < 0)
    isCompressed[idx] = -1 * isCompressed[idx] ;
  else
    isCompressed[idx] = blockSize ;
  }

 __syncthreads() ;

   int i = 1 ;
   localVal = 0 ;
   while ( i < numBlocks)
  {
    int temp ;
    if((idx < numBlocks) && idx >= i && tx == 0)
  {
     localVal += isCompressed[idx - i] ;
     temp = isCompressed[idx] + isCompressed[idx - i] ;
  }
    __syncthreads() ;

    if((idx < numBlocks) && idx >=i && tx ==0)
       isCompressed[idx] = temp ;

    __syncthreads() ;
   i = i*2 ;
 }

  return localVal ;
}*/

//int x = ceil(numElements/1024.0);
//dim3 DecomGrid(x, 1, 1);
//dim3 DecomBlock(1024, 1, 1);
//decompress_n(devCompressed, devDecompressed, devBlockStart, devBaseVals, numElements);
__global__ void decompress_n(char* input, long* output, int16_t* devCompressedTable, int* blockStart, long* blockBase, int numElements){ //, int numElementsperBlock){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    //tx is the Element number
    int blockNum = floorf(tx/(Algo_blockSize/sizeof(long))); //floorf(tx/numElementsperBlock)//Floor rounds down to integer
    //blockStart[blockNum-1] <-- This is starting of a block in Byte index in input array.
    if(tx < numElements){
        int start;
        //if(blockNum == 0)
        //    start = 0;
        //else
            start = blockStart[blockNum];
        int16_t curr_size = devCompressedTable[blockNum];
        long temp;
        int IDinBlock; //Location of current Element within compressed Block
        if(curr_size == 5 or curr_size == 9 or curr_size ==13){
            IDinBlock = tx % (Algo_blockSize /sizeof(long)); //<- Assuming always long at start
            start = start+IDinBlock;
            memcpy(&temp, &input[start], sizeof(uint8_t));
            output[tx] = temp + blockBase[blockNum];
            //memcpy(&output[tx], &input[start], sizeof(uint8_t));
            //output[tx] = (long)((uint8_t) input[start]);// + (uint8_t) blockBase[blockNum]);
        }
        else if(curr_size == 6 or curr_size == 10 or curr_size ==14){
            IDinBlock = tx % (Algo_blockSize /sizeof(long)); //<- Assuming always long at start
            start = start+IDinBlock;
            //output[tx] = (long) start;
            memcpy(&temp, &input[start], sizeof(uint16_t));
            output[tx] = temp + blockBase[blockNum];
            //output[tx] = (long)((uint16_t) input[start]);// + (uint16_t) blockBase[blockNum]);
        }
        else if(curr_size == 7 or curr_size == 11 or curr_size ==15){
            IDinBlock = tx % (Algo_blockSize /sizeof(long)); //<- Assuming always long at start
            memcpy(&temp, &input[start], sizeof(uint32_t));
            output[tx] = temp + blockBase[blockNum];
            //output[tx] = (long)((uint32_t) input[start]);// + (uint32_t) blockBase[blockNum]);
        }
        else if (curr_size <0){
            memcpy(&output[tx], &input[start], -1*curr_size);
        }
    }
}


/*__global__ void decompress_kernel_krishna ( char * decompressed , char * compressed , int numBlocks , int16_t * isCompressed , long * baseVals)
{
   int idx = blockIdx.x ;
   int tx = threadIdx.x ;
   int localVal ;
   __shared__ long blockBaseVal ;
   __shared__ int base_val ;
   if ( idx < numBlocks)
      localVal = isCompressed[idx]  ;
   int localBaseVal = scan_krishna(isCompressed , numBlocks) ;
   if ( tx == 0){
      base_val = localBaseVal;
      blockBaseVal = baseVals[idx] ;
   }
   __syncthreads() ;
   int i ;
   int bytesToDecompress = 1 ;
   int localChunkSize , localCompSize ;
   if ( idx < numBlocks)
   {
      if(localVal > 0)
      {
          localChunkSize = (localVal/4 ) ;
          if(localChunkSize == 1 )
             localChunkSize = 2 ;
          else if (localChunkSize == 2 )
             localChunkSize = 4 ;
          else
             localChunkSize = 8 ;
          localCompSize = localVal % 4 ;
          if(localCompSize == 1)
             localCompSize = 1 ;
          else if (localCompSize == 2)
              localCompSize = 2 ;
          else
              localCompSize = 4 ;
          bytesToDecompress = (blockSize/localChunkSize)*localCompSize ;
       }
      else if ( localVal ==0 )
        bytesToDecompress = blockSize ;
      else
        bytesToDecompress = -1*localVal ;
   }

   __shared__ char compressed_local[blockSize] ;
  if(idx < numBlocks)
  {
   for ( i = 0 ; i < bytesToDecompress ; i += blockDim.x)
   {
     if(i + tx < bytesToDecompress)
        compressed_local[i + tx] = compressed[base_val + i + tx] ;
   }
  }

  __syncthreads() ;
 int baseDecompress = idx * blockSize ;
 int copyGranularity = (localVal<=0) ? 1 : localCompSize ;
 int chunkGranularity = (localVal <=0) ? 1 : localChunkSize ;
 int i1 = baseDecompress ;
 if(idx < numBlocks)
 {
    for(i = 0 ; i < bytesToDecompress  ; i+= copyGranularity * blockDim.x , i1 += blockDim.x * chunkGranularity )
    {
        if (i + copyGranularity * tx < bytesToDecompress)
        {
           if(localVal<=0)
              decompressed[i1 +chunkGranularity*tx] = compressed_local[i + copyGranularity * tx] ;
           else{

                  if (chunkGranularity == 2)
                  {
                    int16_t a = blockBaseVal ;

                    if(copyGranularity == 1)
                    {
                       uint8_t *a1 = (uint8_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else if (copyGranularity ==2)
                    {
                       uint16_t *a1 = (uint16_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else
                    {
                       uint32_t *a1 = (uint32_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                    memcpy (&decompressed[i1 + chunkGranularity * tx] , &a , chunkGranularity) ;
                  }

                   if (chunkGranularity == 4)
                  {
                    int32_t a = blockBaseVal ;

                    if(copyGranularity == 1)
                    {
                       uint8_t *a1 = (uint8_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else if (copyGranularity ==2)
                    {
                       uint16_t *a1 = (uint16_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else
                    {
                       uint32_t *a1 = (uint32_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                    memcpy (&decompressed[i1 + chunkGranularity * tx] , &a , chunkGranularity) ;
                  }
                   if (chunkGranularity == 8)
                  {
                    long a = blockBaseVal ;

                    if(copyGranularity == 1)
                    {
                       uint8_t *a1 = (uint8_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else if (copyGranularity ==2)
                    {
                       uint16_t *a1 = (uint16_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                   else
                    {
                       uint32_t *a1 = (uint32_t*)&(compressed_local[i + copyGranularity * tx]) ;
                       a = a + *a1 ;
                    }
                    memcpy (&decompressed[i1 + chunkGranularity * tx] , &a , chunkGranularity) ;
                  }
               }
           }
      }
   }

  __syncthreads() ;
}*/



int bdCompress(char* input, int len, char * compressed,  int16_t * isCompressed, long * baseVals , int * positions_array)
{

    int bytesCopied = 0 ;
    int blkCounter = 0 ;
    int numBlocks = ceil(len/Algo_blockSize); //int numBlocks = ((len - 1/blockSize)) + 1 ; // ceiling
    int i ;
    int size_index ;
    int size_array[3] ;
    size_array[0] = 16 ; size_array[1] = 32 ; size_array[2] = 64 ;
    int offset = 0 ;

    // long *ptrArray[numPtrs] ;
    if ( offset + Algo_blockSize > len)
    {
        baseVals[blkCounter] = 0 ;
        isCompressed[blkCounter] = offset - len ;
        memcpy(&compressed[bytesCopied] , &input[offset] , len - offset) ;
        positions_array[blkCounter + 1] = positions_array[blkCounter] + (len - offset) ;
        bytesCopied += len - offset ;
        return bytesCopied ;
    }


    while(offset + Algo_blockSize <= len) // Don't want to compress if length remaining is less than block
    {

        // Assume each value is 4 bytes , long , can try to vary this later.
        // Try to compress it to unsigned int_8 (0-255)
        // Get the minimum value as base so unsigned int 8 can be used for deltas
        // If value ranges are more than max value of unsigned int 8 , try to convert to unsigned int.
        // If ranges are more than unsigned int , don't compress this block , move onto next block.
	       //	bool used = false ;
	       //
        int minBytesUsed = -1 ;
        int minCompressed ;
        int minDivision ;
        long minVal ;
        for ( size_index = 0 ; size_index < 3 ; size_index++)
        {
            int base_size = size_array[size_index] ;
            int numPtrs =  (Algo_blockSize*8)/(size_array[size_index]) ; //this is number of elements in block that will be compressed
            long ptrArray [numPtrs] ;
            char local_storage[Algo_blockSize] ;

            for ( i = 0 ; i < numPtrs ; i++)
            {
                ptrArray[i] = getVals((char*)&input[offset + i*(base_size/8)] , (base_size/8)) ; //get input as int16, int32 or long
            }
            bool flag = false ;
            long minValue ;
            for ( i =0 ; i<numPtrs ; i++)
            {
                //We are trying to find minvalue in Block
                //minValue will equal ptrArray[0] at start, then it will become minimum of ptr[i] values
                if(flag==false)
                {
                    flag = true ;
                    minValue = ptrArray[i];
                }
                else
                {
                    if( ptrArray[i] < minValue)
                    minValue = ptrArray[i] ;
                }
            }

            long range = 0 ;
            flag = false ;
            for (i =0 ; i<numPtrs ; i++)
            {
                //After loop, range will equal largest (ptrArray[i] - minValue)
                if(flag ==false)
                {
                    range = (ptrArray[i])  - minValue ;
                    flag = true ;
                }
                else
                {
                    if((ptrArray[i]) - minValue > range)
                    range = ptrArray[i] - minValue ;
                    //  printf("Calc-Range:%ld\n",range) ;
                }
            }
            //printf("Final range is %ld , Size is %d, Min Val is %ld , Num ptrs is %d\n",range , size_array[size_index] , minValue,numPtrs);
            if((range < pow(2 , sizeof(uint8_t) * 8)))
            // compress into uint8
            {
                if(minBytesUsed == -1 ){
                    minBytesUsed = 8*numPtrs ;
                    minCompressed = 8 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                }
                else if ( 8 * numPtrs < minBytesUsed)
                {
                    minBytesUsed = 8 * numPtrs ;
                    minCompressed = 8 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                    //printf("8: Range : %ld , minValue : %ld , min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
            }
            else if ((range < pow(2,sizeof(uint16_t)*8)) && (base_size > 16))
            // compress into uint16
            {
                if(minBytesUsed == -1 ){
                    minBytesUsed = 16 *numPtrs ;
                    minCompressed = 16 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                }
                else if ( 16 * numPtrs < minBytesUsed)
                {
                    minBytesUsed = 16 * numPtrs ;
                    minCompressed = 16 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                    //   printf("16 : Range : %ld, minValue : %ld , min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
            }
            else if ( (range < pow ( 2 , sizeof(uint32_t)*8)) && (base_size > 32))
            // compress into uint32
            {
                if(minBytesUsed == -1 ){
                    minBytesUsed = 32 *numPtrs ;
                    minCompressed = 32 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                }
                else if ( 32 * numPtrs < minBytesUsed)
                {
                    minBytesUsed = 32 * numPtrs ;
                    minCompressed = 32 ;
                    minDivision = base_size ;
                    minVal = minValue ;
                    //  printf("32: Range : %ld, minValue : %ld, min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
            }
            // printf("Bytes used so far :%d\n" , minBytesUsed) ;
        }
        if ( minBytesUsed >= 0 )
        {
            int i ;
            int numPtrs = (Algo_blockSize*8/minDivision) ;
            long  ptrArray[numPtrs] ;
            int div_off , compressed_off ;
            for (i =0 ; i < numPtrs ; i++)
            {

                ptrArray[i] = getVals(&input[offset + (i*minDivision/8)] , (minDivision/8))  ;
                div_off = 3 ;
                if ( minDivision == 16)         //Was unit16 BEFORE compression
                {
                    //   ptrArray[i] = (int16_t *)(&input[offset + i*minDivision]) ;
                    div_off = 1 ;
                }
                else  if (minDivision == 32)    //Was uint32 BEFORE compression
                {
                    //  ptrArray[i] = (int32_t*) (&input[offset + i*minDivision]) ;
                    div_off = 2 ;
                }

                if ( minCompressed == 32)       //is uint32 AFTER compression
                {
                    uint32_t a  = ptrArray[i] - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint32_t)) ;
                    compressed_off = 3 ;
                }
                else if ( minCompressed == 16)  //is uint16 AFTER compression
                {
                    uint16_t a  = ptrArray[i] - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint16_t)) ;
                    compressed_off = 2 ;
                }
                else if ( minCompressed == 8)   //is uint8 AFTER compression
                {
                    uint8_t a  = ptrArray[i] - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint8_t)) ;
                    compressed_off = 1 ;
                }
                bytesCopied += (minCompressed/8) ;


            }
            //isCompressed is 5 if uint16 was compressed into uint8
            //                6 if uint16 was compressed into uint16
            //                7 if uint16 was compressed into uint32
            //                9 if uint32 was compressed into uint8
            //               10 if uint32 was compressed into uint16
            //               11 if uint32 was compressed into uint32
            //               13 if long was compressed into uint8
            //               14 if long was compressed into uint16
            //               15 if long was compressed into uint32
            isCompressed[blkCounter] = 4*div_off + compressed_off ;
            positions_array[blkCounter + 1] = positions_array[blkCounter] + ((Algo_blockSize*8)/minDivision)*(minCompressed/8) ;
            baseVals[blkCounter] = minVal ;
        }
        else
        {
            baseVals[blkCounter] = minVal ;
            isCompressed[blkCounter] = 0 ;
            positions_array[blkCounter + 1] = positions_array[blkCounter] + Algo_blockSize ;
            memcpy(&compressed[bytesCopied] , &input[offset] , Algo_blockSize) ;
            bytesCopied += Algo_blockSize ;
        }
        offset += Algo_blockSize ;
        blkCounter++ ;
        if ( offset + Algo_blockSize > len)
        {
            baseVals[blkCounter] = 0 ;

            isCompressed[blkCounter] = offset - len ;
            positions_array[blkCounter + 1]  = positions_array[blkCounter] + (len - offset) ;
            memcpy(&compressed[bytesCopied] , &input[offset] , len - offset) ;
            bytesCopied += len - offset ;
            break ;
        }

    }

    return bytesCopied ;

}

/*int decompress ( char * compressed , char * decompressed , int bytesCopied , long *baseVals , int16_t *isCompressed , int numBlocks)
{
    int i = 0;
    int offset_compressed = 0;
    int offset_decompressed = 0;
    for (i = 0; i <numBlocks; i++) // decompress every block
    {
        if(isCompressed[i] ==0)
        {
           memcpy(&decompressed[offset_decompressed] , &compressed[offset_compressed] , Algo_blockSize) ;
           offset_compressed += Algo_blockSize ;
           offset_decompressed += Algo_blockSize ;
           continue ;
         }

         if (isCompressed[i] <0)
         {
            int bytes_to_copy = -1* isCompressed[i] ;
            memcpy(&decompressed[offset_decompressed] , &compressed[offset_compressed] , bytes_to_copy) ;
            offset_compressed += bytes_to_copy ;
            offset_decompressed += bytes_to_copy ;
            break ;
         }

         // If code reaches this point then actual compression has taken place
        int chunk_size ;
        int compressed_size ;
        if((isCompressed[i] / 4 ) ==1 )
           chunk_size = 2  ;
        else if ((isCompressed[i]/4) ==2)
           chunk_size = 4 ;
        else
           chunk_size = 8 ;
        if(isCompressed[i] % 4==1)
           compressed_size = 1 ;
        else if (isCompressed[i]%4 ==2)
           compressed_size = 2 ;
        else
           compressed_size = 4 ;

        int numPtrs = Algo_blockSize/chunk_size ;
        int j ;
        for (j = 0 ; j < numPtrs ; ++j)
        {
            long compressed_val = getVals((char *)&compressed[offset_compressed], compressed_size * 8) ;

            compressed_val += baseVals[i] ;
            if (chunk_size ==2 )
            {
              int16_t num = compressed_val ;
              memcpy(&decompressed[offset_decompressed] , &num , chunk_size) ;
            }
           else if (chunk_size ==4)
            {
               int32_t num = compressed_val ;
               memcpy(&decompressed[offset_decompressed] , &num , chunk_size) ;
            }
           else
            {
                long num = compressed_val ;
                memcpy(&decompressed[offset_decompressed] , &num , chunk_size) ;
             }
           offset_compressed += compressed_size ;
           offset_decompressed += chunk_size ;
         }
      }

    return offset_decompressed ;
}*/











int main(int argc, char **argv) {
    //check_success(cudaProfilerStart());
    if (argc !=2 ){
        cerr << "This program perfoms Compression on CPU and Decompression on GPU\n"
        << "Load file to compress as input argument\n"
        << "Sample usage: \n"
        << argv[0]
        << " input512.raw\n";
        return -1;
    }

    //--------------LOAD INPUT FILE-----------------
    //--- 1st row of file has size of input file ---
    //--- Contents start from 2nd row --------------
    int i=0;
    string filename = string(argv[1]);
    string line;
    int longArraySize; //=512
    std::fstream file_in(filename);
    if (file_in.is_open()) {
        //while( std::getline (file_in,line))
        //    ++longArraySize;
        file_in >> longArraySize;
    }
    printf("total lines = %d\n", longArraySize);
    long * inputArray = (long*) malloc(longArraySize*sizeof(long));
    //long inputArray[longArraySize];
    if (file_in.is_open()) {
        for( int i =0; i < longArraySize; i++) {
            //while ( std::getline (file_in,line) ){
            //inputArray[i] = stol(line);
            file_in >> inputArray[i];
            //printf("val=%lu, %d\n", inputArray[i], i);
        }
        file_in.close();
    }
    else {
        cout << "Unable to open file: " << argv[1] << "\n";
        return -1;
    }

    /*ofstream file_out;
     file_out.open(filename);
     for ( i =0 ; i < longArraySize ; i++){
     inputArray[i] = (100*i);
     printf("%lu\n",inputArray[i]);
     file_out << inputArray[i];
     file_out <<"\n";
     }
     file_out.close(); */
    //-----------------Input LOADED -------------

    int numElements = longArraySize;
    int numBytesBeforeCompress = longArraySize*sizeof(long);
    int numBlocks = ceil(numBytesBeforeCompress/Algo_blockSize); //int numBlocks = (((longArraySize * sizeof(long))-1)/Algo_blockSize) + 1 ; //ceiling

    long baseVals[numBlocks] ;
    int16_t  isCompressed[numBlocks] ;
    int positions_array[numBlocks + 1] ;
    char * compressed = (char*) malloc(numBytesBeforeCompress);
    //char* compressed = new char[numBytesBeforeCompress] ; //Compressed table should be big enough as uncompressed data
    initArray(numBlocks, isCompressed) ;
    const auto start = now() ;
    positions_array[0] = 0 ;
    int bytesCopied = bdCompress((char*)inputArray, numBytesBeforeCompress, compressed, isCompressed, baseVals, positions_array) ;
    int numBytesAfterCompress = bytesCopied;
    const auto stop = now() ;
    const auto elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "Time to compress on CPU = " << elapsed << " milliseconds.\n";
    //printf("Length , Bytes copied  : %d , %d\n", numBytesBeforeCompress, bytesCopied) ;

    float compression_ratio = (float)(numBytesBeforeCompress)/(float)(numBlocks*(sizeof(long) + sizeof(int16_t)) + bytesCopied) ;
    for ( i = 0 ; i < numBlocks ; i++){
      //printf("Base value=%lu , compressed info=%d , Ratio=%f\n", baseVals[i] , isCompressed[i] , compression_ratio) ;
    }

    char * Decompressed = (char*) malloc(numBytesBeforeCompress) ;
    //char* Decompressed = new char[numBytesBeforeCompress] ;
    /*//int bytes = decompress(compressed , decompressed , bytesCopied , baseVals , isCompressed , numBlocks) ;
     printf("Bytes after decompression : %d\n" , bytes) ;
     bool t = (bytes == longArraySize * sizeof(long)) && (strncmp((char*)inputArray , decompressed , bytes) ==0) ;
     if(t)
     printf("Successful \n") ;
     */

    char* devCompressed ;
    long* devDecompressed ;
    int16_t* devCompressedTable ;
    long* devBaseVals ;
    int * dev_positions_array ;
    int sizeCompressedTable = numBlocks * sizeof(int16_t);
    int sizeBaseVals = numBlocks * sizeof(long);

    float* BlockStart =         (float*) malloc(numBlocks*sizeof(float)); //new int[numBlocks*sizeof(int)];
    //float* Hostscan1output =    (float*) malloc(numBlocks*sizeof(float));
    float* HostsumBlockInput =  (float*) malloc(numBlocks*sizeof(float));
    int num_sumBlockScan = ceil(numBlocks*SCAN_BLOCK_SIZE/2);
    float* HostsumBlockScan =   (float*) malloc(num_sumBlockScan*sizeof(float));

    float* devBlockStart;
    //float* scan1output;
    float* sumBlockInput;
    float* sumBlockScan;

    cudaMalloc(&devCompressed,      numBytesAfterCompress);
    cudaMalloc(&devDecompressed,    numBytesBeforeCompress);
    cudaMalloc(&devCompressedTable, sizeCompressedTable);
    cudaMalloc(&devBaseVals,        sizeBaseVals);
    cudaMalloc(&dev_positions_array , (numBlocks + 1)*sizeof(int)) ;

    //cudaMalloc(&devBlockStart,              numBlocks*sizeof(int));
    cudaMalloc((void**)&devBlockStart,      numBlocks*sizeof(float));
    cudaMalloc((void**)&sumBlockInput,      numBlocks*sizeof(float));
    cudaMalloc((void**)&sumBlockScan,       num_sumBlockScan*sizeof(float));
    cudaMemset(devBlockStart, 0, numBlocks * sizeof(float));
    // -------- Transfer to GPU -----------
    check_success(cudaMemcpy(devCompressed,       compressed,     numBytesAfterCompress,  cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devCompressedTable,  isCompressed,   sizeCompressedTable,    cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devBaseVals,         baseVals,       sizeBaseVals,           cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(dev_positions_array, positions_array , (numBlocks + 1)*sizeof(int) , cudaMemcpyHostToDevice));

    // get elapsed time in milliseconds
    // const auto elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
    // See this in NVVP profiler. std::cout << "Transfer CPU to GPU time = " << elapsed << " milliseconds.";

    // Decompress in GPU
    // ----------------------------------------
    int x = ceil(numBlocks/(2*SCAN_BLOCK_SIZE+0.0f));
    dim3 ScanGrid(x,1,1);
    dim3 ScanBlock(SCAN_BLOCK_SIZE,1,1);
    dim3 DoubleBlock(SCAN_BLOCK_SIZE*2,1,1);
    //scan_n<<<ScanGrid, ScanBlock>>>( devCompressedTable, devBlockStart, numBlocks); //<- change to exclusive scan
    //cudaDeviceSynchronize();
    //ExtractLast<<<ScanGrid, ScanBlock>>>(devBlockStart, sumBlockInput, numBlocks);
    //scan_regular<<<ScanGrid, ScanBlock>>>(sumBlockInput, sumBlockScan, x);
    //Finaladd<<<ScanGrid, DoubleBlock>>>(sumBlockScan, devBlockStart, numBlocks);
    check_success(cudaDeviceSynchronize());
    //Now devBlockStart will have starting location for each block in Compressed Array

    //for ( i=0; i < numBlocks; i++)
    //    printf("ScanBefore:%d\n", (int)BlockStart[i]);
    //check_success(cudaMemcpy(Hostscan1output,   scan1output,     numBlocks*sizeof(float),  cudaMemcpyDeviceToHost));
    //check_success(cudaMemcpy(HostsumBlockInput, sumBlockInput,   numBlocks*sizeof(float),  cudaMemcpyDeviceToHost));
    //check_success(cudaMemcpy(HostsumBlockScan,  sumBlockScan,    num_sumBlockScan*sizeof(float),        cudaMemcpyDeviceToHost));
    //check_success(cudaMemcpy(BlockStart,        devBlockStart,   numBlocks*sizeof(float),    cudaMemcpyDeviceToHost)) ;
    //check_success(cudaDeviceSynchronize());
    //for ( i=0; i < numBlocks; i++)
    //    printf("BlockStart%d=%f\n", i, BlockStart[i]);
    //for ( i=0; i < numBlocks; i++)
    //    printf("Scan2:%f\n", HostsumBlockInput[i]);
    //for ( i=0; i < num_sumBlockScan; i++)
    //    printf("Scan3:%f\n", HostsumBlockScan[i]);
    //for ( i=0; i < numBlocks; i++)
    //    printf("Scan4:%d\n", BlockStart[i]);

    x = ceil(numElements/1024.0);
    dim3 DecomGrid(x, 1, 1);
    dim3 DecomBlock(1024, 1, 1);
    decompress_n<<<DecomGrid, DecomBlock>>>(devCompressed, devDecompressed, devCompressedTable, dev_positions_array, devBaseVals, numElements); //, numElementsperBlock);
    //decompress_n<<<DecomGrid, DecomBlock>>>(devCompressed, devDecompressed, devCompressedTable, devBlockStart, devBaseVals, numElements); //, numElementsperBlock);
    //dim3 kGrid(numBlocks, 1, 1);
    //dim3 kBlock(blockSize, 1, 1);
    //decompress_kernel_krishna<<<kGrid, kBlock>>>( (char*)devDecompressed, devCompressed, numBlocks, devCompressedTable, devBaseVals);
    cudaDeviceSynchronize();


    //int* newdebug = new int[numBlocks*sizeof(int)];
    //cudaMemcpy(newdebug,   devCompressedTable,sizeCompressedTable,    cudaMemcpyDeviceToHost) ;
    //cudaMemcpy(BlockStart,   devBlockStart,     numBlocks*sizeof(float),cudaMemcpyDeviceToHost) ;
    cudaMemcpy(Decompressed, devDecompressed,   numBytesBeforeCompress, cudaMemcpyDeviceToHost) ;
    // get elapsed time in milliseconds
    // elapsed = std::chrono::duration<double, std::milli>(end_decompress - start_decompress).count();
    // std::cout << "De-Compression in GPU time = " << elapsed << " milliseconds.";
    cudaDeviceSynchronize();
    delete[] compressed;
    // Free Device Memory
    // ----------------------------------------
    //check_success(cudaProfilerStop());

    printf("numBlocks = %d\n",numBlocks);
    printf("numBytesBeforeCompress = %d\n", numBytesBeforeCompress);
    printf("numElements = %d\n", numElements);
    printf("numBytesAfterCompress = %d\n", numBytesAfterCompress);
    printf("Sizeofuint16 = %d\n", sizeof(uint16_t));
    //for ( i = 0 ; i< numBlocks ; i++)
        //printf("IsCompressed:%d\n" , (int)isCompressed[i] ) ;
    //for ( i=0; i < numBlocks; i++)
    //    printf("from GPU:%d\n", newdebug[i]);
    //for ( i=0; i < numBlocks; i++)
        //printf("Scan2nd:%d\n", (int)BlockStart[i]);
    long * l_array = (long*)&Decompressed[0] ;
    //for (i = 0 ; i < longArraySize ; i++)
    //  printf("Dec:%ld     in %d\n" , l_array[i], i) ;


    char* inputA = (char*)&inputArray[0];
    for (i =0; i < numBytesBeforeCompress; i++)
    {
      if (inputA[i] != Decompressed[i])
        printf("MISMATCH at index %d, Input=%c Decompress=%c\n", i, inputA[i], Decompressed[i]);
    }
    /*for (i =0; i < numBlocks; i++)
    {
      printf("left is %f, right is %d\n", BlockStart[i], positions_array[i]);
    }*/
    //int k = strncmp((char*)inputArray, Decompressed, numBytesBeforeCompress) ;
    //if(k==0) cout<<"SUCCESSFUL\n" ;
    cudaFree(devCompressed);
    cudaFree(devDecompressed);
    cudaFree(devCompressedTable);
    cudaFree(devBlockStart);
    cudaFree(devBaseVals);

    return 0;
}
