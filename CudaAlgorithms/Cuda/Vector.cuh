#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float Dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ __forceinline__ float2 Sub(float2 a, float2 b) {
    return { a.x - b.x, a.y - b.y };
}

__device__ __forceinline__ float2 Add(float2 a, float2 b) {
    return { a.x + b.x, a.y + b.y };
}

__device__ __forceinline__ float2 Mul(float2 a, float2 b) {
    return { a.x * b.x, a.y * b.y };
}

__device__ __forceinline__ float2 MulByConst(float2 a, float c) {
    return { a.x * c, a.y * c };
}

__device__ __forceinline__ float2 Normalize(float2 vec) {
    float len2 = vec.x * vec.x + vec.y * vec.y;

    if (len2 < 1e-10f) {
        return { 0, 0 };
    }

    float invLength = rsqrtf(vec.x * vec.x + vec.y * vec.y);
    return { vec.x * invLength, vec.y * invLength };
}
