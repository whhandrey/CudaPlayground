#include "CudaDeviceMemory.h"
#include <cuda_runtime.h>
#include <Cuda/CudaCheck.h>

namespace cuda::memory {
	LinearDeviceMemory::~LinearDeviceMemory() {
		Release();
	}

	void LinearDeviceMemory::Allocate(size_t sizeBytes) {
		Release();

		cudaCheck(cudaMalloc(&m_mem, sizeBytes));
		m_sizeBytes = sizeBytes;
	}

	void LinearDeviceMemory::Release() {
		if (m_mem != nullptr) {
			cudaFree(m_mem);

			m_mem = nullptr;
			m_sizeBytes = 0;
		}
	}

	void* LinearDeviceMemory::Mem() const {
		return m_mem;
	}

	size_t LinearDeviceMemory::SizeBytes() const {
		return m_sizeBytes;
	}
}
