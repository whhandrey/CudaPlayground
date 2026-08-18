#include "CudaDeviceMemory.h"
#include <cuda_runtime.h>
#include <Cuda/CudaCheck.h>

namespace cuda::memory {
	LinearDeviceMemory::~LinearDeviceMemory() {
		Release();
	}

	LinearDeviceMemory::LinearDeviceMemory(LinearDeviceMemory&& other) noexcept
		: m_mem{ std::exchange(other.m_mem, nullptr) }
		, m_sizeBytes{ std::exchange(other.m_sizeBytes, 0) }
	{
	}

	LinearDeviceMemory& LinearDeviceMemory::operator=(LinearDeviceMemory&& other) noexcept {
		if (this != &other) {
			Release();

			m_mem = std::exchange(other.m_mem, nullptr);
			m_sizeBytes = std::exchange(other.m_sizeBytes, 0);
		}

		return *this;
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
