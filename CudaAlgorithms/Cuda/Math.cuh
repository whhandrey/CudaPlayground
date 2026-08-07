#pragma once
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T clamp(T val, T min_val, T max_val) {
    return min(max(val, min_val), max_val);
}

__device__ __forceinline__ float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}
