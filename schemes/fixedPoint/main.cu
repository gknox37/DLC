#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include "fixedPointCompress.h"
#include <cuda_profiler_api.h>

#define M_SIZE 100000
#define RANDO_MIN  -1
#define RANDO_MAX  1
#define BATCH_SIZE 10
#define BATCHES_PER_BLOCK 10

#define ITER 1
#define NOT_COMP 0
#define COMP 1
#define TEST_MAT 1


//each thread processes a block
__global__ void decompress_fixed24_8_gpu(uint8_t* in, unsigned* pointers, unsigned len, fixed_point24_8* out, uint32_t batchSize, int numBatches) {
	__shared__ uint8_t schemes[BATCHES_PER_BLOCK];
	__shared__ unsigned startPos[BATCHES_PER_BLOCK];

    int idx = threadIdx.x + blockIdx.x*blockDim.x;	
    int batchIdx = idx % batchSize;
    int myBatch = ((float)idx/(float)len)* numBatches;
	int localBatchNum = myBatch%BATCHES_PER_BLOCK;
	int myNum;

	//rep thread gets compression scheme
	if(batchIdx == 0){
		startPos[localBatchNum] = pointers[myBatch];
    	schemes[localBatchNum] = in[startPos[localBatchNum]];
	}
	__syncthreads();
	
    //copying results
	//if (idx < len){
	//	memcpy(&myNum, &in[startPos[localBatchNum] + 1 + 2*batchIdx], 2);
    //	out[idx].data = (int16_t)(((myNum&0xffffff00) >> 8)   |  ((myNum & 0x000000ff) <<8));
	//}
	if(idx < len){
		out[idx].data = myNum;
	}
	
}

__global__ void decompress_fixed24_8_gpu_dummy(uint8_t* in, unsigned len, fixed_point24_8* out, uint32_t batchSize) {
  return;
}


int main() {
    /*Allocations and declarations*/
    //Host
	bool match;
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
	cudaProfilerStart();
    //for(int count = 0; count < ITER*2; count ++){
        //mode = count %2;
	mode = 1;
    
    /*Allocating host space for data */
    in = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
    in_decompressed = (fixed_point24_8*) malloc(sizeof(fixed_point24_8)*M_SIZE);
	//memset(in_decompressed,0,sizeof(fixed_point24_8)*M_SIZE);
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
		printf("Beginning Compression Mode\n");
		/*Beginning cpu timer  */
		begin = clock();
		numBytes = compressFixed24_8(in,M_SIZE,0,BATCH_SIZE,in_compressed,pointers);
		end = clock();    
		timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
		printf("Compression Time: %f ms\n", timeSpent);
		begin = clock();
		
		/*Copying host to device*/
		printf("Number of bytes to copy = %d\n",numBytes);
		printf("Number of batches = %d\n",numBatches);
		cudaMemcpy(in_compressed_D, in_compressed, numBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(pointers_D, pointers, numBatches*sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemset(out_decompressed_D,0, M_SIZE*sizeof(fixed_point24_8)); //TODO Check if writing output array is necessary
		//TODO Check if writing output array is necessary

		/*Launching kernel*/
		xWidth = BATCH_SIZE * BATCHES_PER_BLOCK; yWidth =1;
		numXBlocks = ceil((float)M_SIZE/(float)xWidth); numYBlocks = 1;
		printf("xWidth = %d\n",xWidth);
		printf("numXBlocks = %d\n",numXBlocks);
		dim3 dimGrid(numXBlocks, numYBlocks,1);
		dim3 dimBlock(xWidth, yWidth,1);
		decompress_fixed24_8_gpu<<<dimGrid,dimBlock>>>(in_compressed_D, pointers_D, M_SIZE, out_decompressed_D, BATCH_SIZE, numBatches); //TODO SIZE_M used to be cudaDeviceSynchronize();

		/*Ending Timer*/
		end = clock();    
		timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
		printf("Compressed Kernel: %f ms\n", timeSpent);
    }
    else if(mode == NOT_COMP){
		printf("Beginning UnCompressed Mode\n");
        /*Beginning cpu timer  */
	    begin = clock();
        
        /*Copying host to device*/
        cudaMemcpy(in_compressed_D, in, M_SIZE*sizeof(fixed_point24_8), cudaMemcpyHostToDevice); // remember this is the uncompressed array
		cudaMemset(out_decompressed_D,0, M_SIZE*sizeof(fixed_point24_8)); //TODO Check if writing output array is necessary

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
    cudaMemcpy(in_decompressed, out_decompressed_D , M_SIZE*sizeof(fixed_point24_8), cudaMemcpyDeviceToHost);

    /*Checking valid decompressed data*/
    match = 1;
    for(int i =0; i < M_SIZE && TEST_MAT == 1; i++){
        if(in_decompressed[i].data != in[i].data){
            //printf("i=%d|Difference with %x and %x\n",i,in_decompressed[i].data, in[i].data);
			//printf("i=%d|Value %d\n",i,in_decompressed[i].data);
            match = 0;
			if(i>50){}
				//break; 
        }
    }
    if(match){
        printf("Matricies match\n");
    }
    cudaProfilerStop();
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
    //}
    printf("Finished\n");
    return 0;
}
