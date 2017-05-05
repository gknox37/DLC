#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <fstream> 
#include <time.h>
#include "fixedPointCompress.h"
#include <cuda_profiler_api.h>

//#define M_SIZE 1048576
#define RANDO_MIN  -1
#define RANDO_MAX  1
#define BATCH_SIZE 64 //Compressed elements per batch
#define BATCHES_PER_BLOCK 8

#define NOT_COMP 0
#define COMP 1
#define TEST_MAT 1


//each thread processes a block
__global__ void decompress_fixed24_8_gpu(uint8_t* in, unsigned* pointers, unsigned len, fixed_point24_8* out, uint32_t batchSize, int numBatches) {
	
	__shared__ uint8_t schemes[BATCHES_PER_BLOCK];
	__shared__ unsigned startPos[BATCHES_PER_BLOCK];

  int idx = threadIdx.x + blockIdx.x*blockDim.x;	
  if (idx < len){
    int batchIdx = idx % batchSize;
    int myBatch = ((float)idx/(float)len)* numBatches;
	  int localBatchNum = myBatch%BATCHES_PER_BLOCK; //perBlock
	
	
	  //rep thread gets compression scheme
	  if(batchIdx == 0){
		  startPos[localBatchNum] = pointers[myBatch];
      	//schemes[localBatchNum] = in[startPos[localBatchNum]]; //TODO BREAKS
	  }
	  __syncthreads();
	
      //copying results
	  unsigned myStart = startPos[localBatchNum];	
	  out[idx].data = (int) (int16_t)(in[myStart + 1 + 2*batchIdx] << 8 | in[myStart + 1 + 2*batchIdx + 1]);
  }
	
}

__global__ void decompress_fixed24_8_gpu_dummy(uint8_t* in, unsigned len, fixed_point24_8* out, uint32_t batchSize) {
  return;
}


int run(int size, int moder) {
  //Defnies
	int M_SIZE = pow(2, size);
	int mode = moder;


  /*Allocations and declarations*/
  //Host
	bool match;
  int numBytes;
  int numXBlocks,numYBlocks,xWidth,yWidth;
  double timeSpent;
  clock_t begin, end;
  fixed_point24_8* in;
  fixed_point24_8* in_decompressed;
  uint8_t* in_compressed;
  unsigned* pointers;
  int bytes;
  //Device
  uint8_t* in_compressed_D;
  unsigned* pointers_D;
  fixed_point24_8* out_decompressed_D;
  //static vars
  int numBatches = ceil((float)M_SIZE / (float)BATCH_SIZE);
  int worstCaseBytes = M_SIZE*(sizeof(fixed_point24_8) + 1) + numBatches*sizeof(unsigned);
	cudaProfilerStart();
    
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
		//printf("Beginning Compression Mode\n");
		/*Beginning cpu timer  */
		begin = clock();
		numBytes = compressFixed24_8(in,M_SIZE,0,BATCH_SIZE,in_compressed,pointers);
		end = clock();    
		timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
		//printf("Compression Time: %f ms\n", timeSpent);
		begin = clock();

		
		//looking at pointers
		/*
		for( int i = 0; i< numBatches; i++){
			printf("BatchNumber = %d | pointer = %d\n",i,pointers[i]);
		}	
		return 0;	
		*/

		//comparing matricies
		/*
		int retVal = decompressFixed24_8(in_compressed, pointers, M_SIZE, in_decompressed, BATCH_SIZE);
		for(int i = 0; i < M_SIZE; i++){
		    int sub = in[i].data - in_decompressed[i].data; 
		    //printf("i=%d| %d - %d = %d\n", i, in[i].data, in_decompressed[i].data, sub);
		    if(sub != 0){
		        printf("ERROROROROROR\n");
		        return -1;
		    }
		}
		printf("Matricies match for cpu!!!\n");	
		return;
		//*/
		
		
		/*Copying host to device*/
		//printf("Number of bytes to copy = %d\n",numBytes);
		//printf("Number of batches = %d\n",numBatches);
		cudaMemcpy(in_compressed_D, in_compressed, numBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(pointers_D, pointers, numBatches*sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemset(out_decompressed_D,0, M_SIZE*sizeof(fixed_point24_8)); 

		/*Launching kernel*/
		xWidth = BATCH_SIZE * BATCHES_PER_BLOCK; yWidth =1;
		numXBlocks = ceil((float)M_SIZE/(float)xWidth); numYBlocks = 1;
		//printf("xWidth = %d\n",xWidth);
		//printf("numXBlocks = %d\n",numXBlocks);
		decompress_fixed24_8_gpu<<<numXBlocks,xWidth>>>(in_compressed_D, pointers_D, M_SIZE, out_decompressed_D, BATCH_SIZE, numBatches); 
		cudaDeviceSynchronize();

		/*Ending Timer*/
		end = clock();    
		timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
		bytes = numBytes + sizeof(unsigned)*numBatches;
		printf("Compressed Kernel: %f ms| and occupied %d bytes\n", timeSpent, bytes);
    }
    else if(mode == NOT_COMP){
		  //printf("Beginning UnCompressed Mode\n");
        /*Beginning cpu timer  */
	    begin = clock();
        
        /*Copying host to device*/
    cudaMemcpy(in_compressed_D, in, M_SIZE*sizeof(fixed_point24_8), cudaMemcpyHostToDevice); // remember this is the uncompresbatchIdxsed array
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
        timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
        bytes = M_SIZE*sizeof(fixed_point24_8);
        printf("Total Time(No compression): %f ms | and occupied %d bytes\n", timeSpent, bytes);
    }

    /*Copying memory back*/
    cudaMemcpy(in_decompressed, out_decompressed_D , M_SIZE*sizeof(fixed_point24_8), cudaMemcpyDeviceToHost);
	  cudaDeviceSynchronize();

    /*Checking valid decompressed data*/
    match = 1;
    for(int i =0; i < M_SIZE && TEST_MAT == 1; i++){
        if(in_decompressed[i].data != in[i].data){
            //printf("MSIZE is %d\n", M_SIZE);
            //printf("i=%d|Difference with %x and %x\n",i,in_decompressed[i].data, in[i].data);
			      //printf("i=%d|Value %d\n",i,in_decompressed[i].data);
            match = 0;
        }
    }
    if(match){
      printf("Matricies match\n");
    }
    else{
      printf("Dont Match\n");
    }
    
    //Writing output to a file
    if (mode == COMP){
      //FILE* f = fopen("out_comp.csv", "a+");
      //printf(f,"%f,",timeSpent);  
      //fclose(f);
      std::ofstream f("out_comp.csv", std::ofstream::out | std::ofstream::app);
      f << timeSpent << ",";
      f.close();
    }
    else{
      std::ofstream f("out_nocomp.csv", std::ofstream::out | std::ofstream::app);
      f << timeSpent << ",";
      f.close();
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
    //printf("Finished\n");
    return 0;
}

//0 no compress | 1 compress
int main(){
  
  std::ofstream f("out_comp.csv");
  f <<  " Compressed Kernel,";
  f.close();
  std::ofstream f2("out_nocomp.csv");
  f2 <<  " Uncompressed Kernel,";
  f2.close();

  for(int i = 1; i< 25; i++){
    printf("----------------(  %d  )-------------\n",i);
    run(i,0);
    run(i,1);
    
  }
}

