#include <stdio.h>

#ifndef FIXED_POINT_COMPRESS_H
#define FIXED_POINT_COMPRESS_H


uint32_t compressFixed32(uint32_t* in, uint32_t len, uint8_t precision, uint8_t loss, uint32_t batchSize, uint32_t* out, uint32_t* pointers);


//found 
// http://stackoverflow.com/questions/187713/converting-floating-point-to-fixed-point
template <class BaseType, size_t FracDigits>
class fixed_point
{
    const static BaseType factor = 1 << FracDigits;

    BaseType data;

public:
    fixed_point(double d)
    {
        *this = d; // calls operator=
    }

    fixed_point& operator=(double d)
    {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    BaseType raw_data() const
    {
        return data;
    }

    // Other operators can be defined here
};


#endif // FIXED_POINT_COMPRESS_
