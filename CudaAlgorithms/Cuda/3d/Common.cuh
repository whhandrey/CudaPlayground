#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ int FlatIdx(int x, int y, int z, int width, int height) {
	return x + y * width + z * width * height;
}

__device__ __forceinline__ size_t OffsetToRowPitched(int y, int z, size_t pitch, size_t slicePitch) {
	return y * pitch + z * slicePitch;
}
