#pragma once
#include <Image/ImageTypes.h>
#include <Cuda/CudaCheck.h>
#include <vector>

namespace common {
	template <class T>
	inline T* AllocAndCopyCPU(const std::vector<T>& vec, cudaStream_t stream) {
		T* vec_gpu = nullptr;

		cudaCheck(cudaMallocAsync(&vec_gpu, vec.size() * sizeof(T), stream));
		cudaCheck(cudaMemcpyAsync(vec_gpu, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

		return vec_gpu;
	}
}
