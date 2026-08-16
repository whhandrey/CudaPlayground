#include "CudaDeviceMemory.h"
#include <cuda_runtime.h>
#include <Cuda/CudaCheck.h>

namespace cuda::memory {
	LinearDeviceMemory::LinearDeviceMemory(size_t sizeBytes)
		: m_sizeBytes{ sizeBytes }
	{
		cudaCheck(cudaMalloc(&m_mem, sizeBytes));
	}

	LinearDeviceMemory::~LinearDeviceMemory() {
		cudaFree(m_mem);
	}

	void* LinearDeviceMemory::Mem() const {
		return m_mem;
	}

	size_t LinearDeviceMemory::SizeBytes() const {
		return m_sizeBytes;
	}
}
