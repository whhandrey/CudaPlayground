#pragma once
#include <Image/ImageTypes.h>
#include <cuda_runtime.h>
#include <sstream>
#include <vector>

#define cudaCheck(err) checkCudaError(err, __FILE__, __LINE__)
inline void checkCudaError(cudaError error, const char* file, const int line) {
	if (error == cudaSuccess)
		return;

	std::ostringstream strm;
	strm << "CudaError: " << cudaGetErrorString(error) << "; File: " << file << "; Line: " << line;

	throw std::runtime_error(strm.str());
}

#define GpuStructAlignSize 16
#define GpuStructAlign __align__(GpuStructAlignSize)

struct range {
	int min;
	int max;
};

inline dim3 DivUp(image::vec2ui dim, image::vec2ui blockSize) {
	return {
		(dim.x + blockSize.x - 1) / blockSize.x,
		(dim.y + blockSize.y - 1) / blockSize.y,
		1
	};
}

inline dim3 Div(image::vec2ui dim, image::vec2ui blockSize) {
	return {
		dim.x / blockSize.x,
		dim.y / blockSize.y,
		1
	};
}

inline dim3 vec2Todim3(image::vec2ui dim) {
	return { dim.x, dim.y, 1 };
}

namespace common {
	template <class T>
	inline T* AllocAndCopyCPU(const std::vector<T>& vec, cudaStream_t stream) {
		T* vec_gpu = nullptr;

		cudaCheck(cudaMallocAsync(&vec_gpu, vec.size() * sizeof(T), stream));
		cudaCheck(cudaMemcpyAsync(vec_gpu, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

		return vec_gpu;
	}
}

namespace util {
	inline std::string BlockDimToString(const image::vec2ui& blockDim) {
		return "(" + std::to_string(blockDim.x) + ", " + std::to_string(blockDim.y) + ")";
	}
}

namespace gpu {
	enum class Arch {
		Unknown,
		Turing,
		Ampere,
		Ada,
		Blackwell
	};

	Arch DetectArch(int device = 0);
}
