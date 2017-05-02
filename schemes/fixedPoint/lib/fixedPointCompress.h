#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef FIXED_POINT_COMPRESS_H
#define FIXED_POINT_COMPRESS_H

#define base_pos 0 
#define base_neg 1
#define base 0
#define mask24_8 0xFFFFFF00
#define mask16_16 0xFFFF0000


//found 
// http://stackoverflow.com/questions/187713/converting-floating-point-to-fixed-point
template <class BaseType, size_t FracDigits>
class fixed_point2
{
    const static BaseType factor = 1 << FracDigits;

    BaseType data;

public:
    fixed_point2(double d)
    {
        *this = d; // calls operator=
    }

    fixed_point2& operator=(double d)
    {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    BaseType raw_data() const
    {
        return data;
    }



    // Other operate667ors can be defined here
};

//static class for 24/8 perciscion
class fixed_point24_8{
public:
    int data;
    fixed_point24_8(float f){
        insert(f);
    }
    
    void insert(float f){
        //TODO add bounds checking 
        float tempf = f*pow(2,8);
        data = (int) tempf;
        /*
        int16_t temp = (int16_t) f;
        float cData = (float)data/ pow(2,8);
        printf("Incoming Float is %f\n",f);
        printf("Fixedpoint looks like %x\n",data);
        printf("Back to floating point is %f\n",cData);
        / */
    }
    float convert(){
        return (float)data/pow(2,8);
    }
};

//Functions
uint32_t compressFixed24_8(fixed_point24_8* in, uint32_t len, uint8_t loss, uint32_t batchSize, uint8_t* out, unsigned* pointers);	
uint32_t decompressFixed24_8(uint8_t* in, unsigned* pointers, unsigned len, fixed_point24_8* out, uint32_t batchSize);

#endif // FIXED_POINT_COMPRESS_
