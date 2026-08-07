#pragma once
#include <cuda_runtime.h>
#include "Math.cuh"

__device__ __forceinline__ unsigned char floatToUchar(float x) {
    x = fminf(fmaxf(x, 0.0f), 255.0f);
    return static_cast<unsigned char>(x + 0.5f);
}

__device__ __forceinline__ uchar4 floatVecToUchar(float4 v) {
    return {
        floatToUchar(v.x),
        floatToUchar(v.y),
        floatToUchar(v.z),
        floatToUchar(v.w)
    };
}

__device__ __forceinline__ unsigned char normFloatToUchar(float x) {
    return static_cast<unsigned char>(saturate(x) * 255.0f + 0.5f);
}

__device__ __forceinline__ int2 float2ToInt2(float2 value) {
    return { __float2int_rn(value.x), __float2int_rn(value.y) };
}
