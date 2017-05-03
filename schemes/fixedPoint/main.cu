#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include "fixedPointCompress.h"

#define M_SIZE 100000
#define RANDO_MIN  -50
#define RANDO_MAX  50
#define BATCH_SIZE 1
#define ITER 2
#define NOT_COMP 0
#define TEST 0
#define COMP 1 

//each thread processes a block
__global__ void decompress_fixed24_8_gpu(uint8_t* in, unsigned* pointers, unsigned len, fixed_point24_8* out, uint32_t batchSize, int numBatches) {
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    //int batchIdx = idx % batchSize;
    //int myBatch = ((float)idx/(float)len)* numBatches;
    unsigned startPos = pointers[idx];
    uint8_t scheme = in[startPos] + 1;
    unsigned myNum;    

    //copying bytes 
    for(int i =0; i< batchSize; i++){
        memcpy(&myNum, &in[startPos + 1 + idx*(scheme)], scheme);
        memcpy(&out[idx*batchSize +i],&myNum,4);
    }
    
}

__global__ void decompress_fixed24_8_gpu_dummy(uint8_t* in, unsigned len, fixed_point24_8* out, uint32_t batchSize) {
  return;
}


int main() {
    /*Allocations and declarations*/
    //Host
    int mode;
    int numBytes;
    int numXBlocks,numYBlocks,xWidth,yWidth;
    double timeSpent;
    clock_t begin, end;
    fixed_point24_8* in;
    fixed_point24_8* in_decompressed;
    uint8_t* in_compressed;
    unsigned* pointers;
    //Device
    uint8_t* in_compressed_D;
    unsigned* pointers_D;
    fixed_point24_8* out_decompressed_D;
    //static vars
    int numBatches = ceil((float)M_SIZE / (float)BATCH_SIZE);
    int worstCaseBytes = M_SIZE*(sizeof(fixed_point24_8) + 1) + numBatches*sizeof(unsigned);

    for(int count = 0; count < ITER*2; count ++){
        mode = count %2;
        
        /*Allocating host space for data */
        in = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
        in_decompressed = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
        in_compressed = (uint8_t*) malloc(worstCaseBytes);
        pointers = (unsigned*) malloc((numBatches) * sizeof(unsigned));
        //creating random values 
        srand( time(NULL) );
        for (int idx  = 0; idx < M_SIZE; idx++){
            float currRand =  RANDO_MIN + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(RANDO_MAX-RANDO_MIN)));
            in[idx].insert(currRand); 
            //printf("Val :%f\n", currRand);
        }

        /*Allocating GPU data arrays*/
        cudaMalloc((void **)&in_compressed_D, worstCaseBytes);
        cudaMalloc((void **)&pointers_D, numBatches*sizeof(unsigned));
        cudaMalloc((void **)&out_decompressed_D, sizeof(fixed_point24_8)*M_SIZE);

        if(mode == COMP){
        /*Beginning cpu timer  */
        begin = clock();
        numBytes = compressFixed24_8(in,M_SIZE,0,BATCH_SIZE,in_compressed,pointers);
        end = clock();    
        timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
        printf("Compression Time: %f ms\n", timeSpent);
        begin = clock();
        
        /*Copying host to device*/
        cudaMemcpy(in_compressed_D, in_compressed, numBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(pointers_D, pointers, numBatches*sizeof(unsigned), cudaMemcpyHostToDevice);
        //TODO Check if writing output array is necessary

        /*Launching kernel*/
        xWidth = 256;yWidth =1;
        numXBlocks = ceil((float)M_SIZE/(float)xWidth);numYBlocks = 1;
        dim3 dimGrid(numXBlocks, numYBlocks,1);
        dim3 dimBlock(xWidth, yWidth,1);
        decompress_fixed24_8_gpu<<<dimGrid,dimBlock>>>(in_compressed_D, pointers_D, numBytes, out_decompressed_D, BATCH_SIZE, numBatches);
        cudaDeviceSynchronize();

        /*Ending Timer*/
        end = clock();    
        timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
        printf("Compressed Kernel: %f ms\n", timeSpent);
        }
        else if(mode == NOT_COMP){
            /*Beginning cpu timer  */
	        begin = clock();
            
            /*Copying host to device*/
            cudaMemcpy(in_compressed_D, in, M_SIZE*sizeof(fixed_point24_8), cudaMemcpyHostToDevice);
            //TODO Check if writing output array is necessary

            /*Launching kernel*/
            numXBlocks = 1;numYBlocks = 1;
            xWidth = 1;yWidth =1;
            dim3 dimGrid(numXBlocks, numYBlocks,1);
            dim3 dimBlock(xWidth, yWidth,1);
            decompress_fixed24_8_gpu_dummy<<<dimGrid,dimBlock>>>(in_compressed_D, numBytes, out_decompressed_D, BATCH_SIZE);
            cudaDeviceSynchronize();

            /*Ending Timer*/
            end = clock();    
            double timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
            printf("Total Time(No compression): %f ms\n", timeSpent);
        }

        /*Copying memory back*/
        cudaMemcpy(in_decompressed, out_decompressed_D , M_SIZE, cudaMemcpyDeviceToHost);

        /*Checking valid decompressed data*/
        bool match = 1;
        for(int i =0; i < M_SIZE && TEST == 1; i++){
            if(in_decompressed[i].data != in[i].data){
                printf("i=%d|Difference with %x and %x\n",i,in_decompressed[i].data, in[i].data);
                match = 0;
            }
        }
        if(match){
            printf("Matricies match\n");
        }
        
        /*Freeing memory*/
        //Host
        free(in);
        free(in_decompressed);
        free(in_compressed);
        free(pointers);
        //Device
        cudaFree(in_compressed_D);
        cudaFree(pointers_D);
        cudaFree(out_decompressed_D);    
    }
    printf("Finished\n");
    return 0;
}
