#pragma once
#include <cuda_runtime.h>
#include <Cuda/CudaCheck.h>

class CudaStream {
public:
	CudaStream() {
		cudaCheck(cudaStreamCreate(&m_stream));
	}

	~CudaStream() {
		cudaCheck(cudaStreamDestroy(m_stream));
	}

	cudaStream_t Get() const {
		return m_stream;
	}

private:
	cudaStream_t m_stream = nullptr;
};
