#pragma once
#include <cuda_runtime.h>
#include <Image/ImageTypes.h>

template <typename T>
__device__ __forceinline__ T clamp(T val, T min_val, T max_val) {
    return min(max(val, min_val), max_val);
}

__device__ __forceinline__ unsigned char floatToUchar(float x) {
    x = fminf(fmaxf(x, 0.0f), 255.0f);
    return static_cast<unsigned char>(x + 0.5f);
}

__device__ __forceinline__ image::vec4uc floatVecToUchar(image::vec4f v) {
    return {
        floatToUchar(v.x),
        floatToUchar(v.y),
        floatToUchar(v.z),
        floatToUchar(v.w)
    };
}
