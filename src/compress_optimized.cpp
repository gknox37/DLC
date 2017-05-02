#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#define blockSize 128 // Power of 2 ?
using namespace std;

long getVals(char * t , int base_size)
{
    long a ;
    if(base_size == 16)
        a = *((int16_t *)t) ;
    else if(base_size == 32)
        a = *((int32_t*)t) ;
    else
        a = *((long*)t) ;
    return a ;
}


void initArray(int size , int16_t *isCompressed)
{
    int i ;
    for ( i =0 ; i < size ; i++)
    {
        isCompressed[i] = 0 ;
    }
}

int bdCompress(char* c , int len , char * compressed ,  int16_t * isCompressed , long * baseVals)
{
    
    int bytesCopied = 0 ;
    int blkCounter = 0 ;
    int numBlocks = ((len - 1/blockSize)) + 1 ; // ceiling
    int i ;
    int size_index ;
    int size_array[3] ;
    size_array[0] = 16 ; size_array[1] = 32 ; size_array[2] = 64 ;
    int offset = 0 ;
    
    // long *ptrArray[numPtrs] ;
    if ( offset + blockSize > len)
    {
        baseVals[blkCounter] = 0 ;
        isCompressed[blkCounter] = offset - len ;
        memcpy(&compressed[bytesCopied] , &c[offset] , len - offset) ;
        bytesCopied += len - offset ;
        return bytesCopied ;
    }
    
    
    while(offset + blockSize <= len) // Don't want to compress if length remaining is less than block
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
            int numPtrs =  (blockSize*8)/(size_array[size_index]) ;
            long ptrArray [numPtrs] ;
            char local_storage[blockSize] ;
            
            for ( i = 0 ; i < numPtrs ; i++)
            {
                ptrArray[i] = getVals((char*)&c[offset + i*(base_size/8)] , base_size) ;
            }
            bool flag = false ;
            long minValue ;
            for ( i =0 ; i<numPtrs ; i++)
            {
                if(flag==false)
                {
                    flag = true ;
                    minValue =  ptrArray[i];
                }
                else
                {
                    if( ptrArray[i]< minValue)
                        minValue = ptrArray[i] ;
                }
            }
            
            long range = 0 ;
            flag = false ;
            for (i =0 ; i<numPtrs ; i++)
            {
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
            printf("Final range is %ld , Size is %d, Min Val is %ld , Num ptrs is %d\n",range , size_array[size_index] , minValue,numPtrs);
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
                    printf("8: Range : %ld , minValue : %ld , min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
                
                
            }
            else if ((range < pow(2,sizeof(uint16_t)*8)) && (base_size > 16))
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
                    printf("16 : Range : %ld, minValue : %ld , min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
                
                
                
                
            }
            
            else if ( (range < pow ( 2 , sizeof(uint32_t)*8)) && (base_size > 32))
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
                    printf("32: Range : %ld, minValue : %ld, min_div:%d , numPtrs:%d\n" , range , minValue,base_size,numPtrs) ;
                }
                
                
                
            }
            printf("Bytes used so far :%d\n" , minBytesUsed) ;
        }
        if ( minBytesUsed >= 0 )
        {
            int i ;
            int numPtrs = (blockSize*8/minDivision) ;
            long  ptrArray[numPtrs] ;
            int div_off , compressed_off ;
            for (i =0 ; i < numPtrs ; i++)
            {
                
                ptrArray[i] = getVals(&c[offset + (i*minDivision/8)] , minDivision)  ;
                div_off = 3 ;
                if ( minDivision == 16)
                {
                    //   ptrArray[i] = (int16_t *)(&c[offset + i*minDivision]) ;
                    div_off = 1 ;
                }
                else  if (minDivision == 32)
                {
                    //  ptrArray[i] = (int32_t*) (&c[offset + i*minDivision]) ;
                    div_off = 2 ;
                }
                
                if ( minCompressed == 32)
                {
                    uint32_t a  = ptrArray[i] - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint32_t)) ;
                    compressed_off = 3 ;
                }
                else if ( minCompressed == 16)
                {
                    uint16_t a  = ptrArray[i] - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint16_t)) ;
                    compressed_off = 2 ;
                    
                    
                }
                else if ( minCompressed == 8)
                {
                    
                    uint8_t a  = (ptrArray[i]) - minVal ;
                    memcpy(&compressed[bytesCopied] , &a , sizeof(uint8_t)) ;
                    compressed_off = 1 ;
                    
                }
                bytesCopied += (minCompressed/8) ;
                
                
            }
            isCompressed[blkCounter] = 4*div_off + compressed_off ;
            baseVals[blkCounter] = minVal ;
        }
        
        else
        {
            baseVals[blkCounter] = minVal ;
            isCompressed[blkCounter] = 0 ;
            memcpy(&compressed[bytesCopied] , &c[offset] , blockSize) ;
            bytesCopied += blockSize ;
        }
        
        
        offset += blockSize ;
        blkCounter ++ ;
        if ( offset + blockSize > len)
        {
            baseVals[blkCounter] = 0 ;
            isCompressed[blkCounter] = offset - len ;
            memcpy(&compressed[bytesCopied] , &c[offset] , len - offset) ;
            bytesCopied += len - offset ;
            break ;
        }
        
    }
    
    return bytesCopied ;
    
}

int decompress ( char * compressed , char * decompressed , int bytesCopied , long *baseVals , int16_t *isCompressed , int numBlocks)
{
    int i = 0 ;
    int offset_compressed = 0 ;
    int offset_decompressed = 0 ;
    for (i = 0 ; i <numBlocks ; i++) // decompress every block
    {
        if(isCompressed[i] ==0)
        {
            memcpy(&decompressed[offset_decompressed] , &compressed[offset_compressed] , blockSize) ;
            offset_compressed += blockSize ;
            offset_decompressed += blockSize ;
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
        
        int numPtrs = blockSize/chunk_size ;
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
}




int main(int argc , char *argv[])
{
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
    int longArraySize;
    std::fstream file_in(filename);
    if (file_in.is_open()) {
        //while( std::getline (file_in,line))
        //    ++longArraySize;
        file_in >> longArraySize;
    }
    printf("total lines = %d\n", longArraySize);
    long testArray[longArraySize];
    if (file_in.is_open()) {
        for( int i =0; i < longArraySize; i++) {
        //while ( std::getline (file_in,line) ){
            //testArray[i] = stol(line);
            file_in >> testArray[i];
            printf("val=%lu, %d\n", testArray[i], i);
        }
        file_in.close();
    }
    else {
        cout << "Unable to open file: " << argv[1] << "\n";
        return -1;
    }
    
    //int longArraySize = 512;
    /*int i;
    ofstream file_out;
    file_out.open(filename);
    for ( i =0 ; i < longArraySize ; i++){
        testArray[i] = (100*i);
        printf("%lu\n",testArray[i]);
        file_out << testArray[i];
        file_out <<"\n";
    }
    file_out.close(); */
    //-----------------Input LOADED -------------
    
    int numBlocks = (((longArraySize * sizeof(long))-1)/blockSize) + 1 ; //ceiling
    
    long baseVals[numBlocks] ;
    int16_t  isCompressed[numBlocks] ;
    char * compressed = (char*) malloc(100000) ;
    initArray(numBlocks , isCompressed) ;
    int bytesCopied = bdCompress((char*)testArray , longArraySize * sizeof(long) ,compressed ,  isCompressed , baseVals) ;
    printf("Length , Bytes copied  : %d , %d\n" , longArraySize*sizeof(long) , bytesCopied) ;
    float compression_ratio = (float)(numBlocks*(sizeof(long) + sizeof(int16_t)) + bytesCopied)/(longArraySize * sizeof(long)) ;
    for ( i = 0 ; i < numBlocks ; i++)
    {
        printf("Base value , compressed info , Ratio : %lu , %d\n , %f\n", baseVals[i] , isCompressed[i] , compression_ratio) ;
    }
    char * decompressed = (char*) malloc(100000) ;
    int bytes = decompress(compressed , decompressed , bytesCopied , baseVals , isCompressed , numBlocks) ;
    printf("Bytes after decompression : %d\n" , bytes) ;
    bool t = (bytes == longArraySize * sizeof(long)) && (strncmp((char*)testArray , decompressed , bytes) ==0) ;
    if(t)
        printf("Successful \n") ;
    
}

