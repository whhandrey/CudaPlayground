#pragma once
#include <sstream>
#include <cuda_runtime.h>

#define cudaCheck(err) checkCudaError(err, __FILE__, __LINE__)
inline void checkCudaError(cudaError error, const char* file, const int line) {
	if (error == cudaSuccess)
		return;

	std::ostringstream strm;
	strm << "CudaError: " << cudaGetErrorString(error) << "; File: " << file << "; Line: " << line;

	throw std::runtime_error(strm.str());
}
