#include "PinnedHostMemory.h"
#include <Cuda/CudaCheck.h>
#include <utility>
#include <cuda_runtime.h>

namespace cuda::memory {
	PinnedHostMemory::PinnedHostMemory(std::size_t sizeBytes) {
		Allocate(sizeBytes);
	}

	PinnedHostMemory::~PinnedHostMemory() {
		Release();
	}

	PinnedHostMemory::PinnedHostMemory(PinnedHostMemory&& other) noexcept
		: m_mem{ std::exchange(other.m_mem, nullptr) }
		, m_sizeBytes{ std::exchange(other.m_sizeBytes, 0) }
	{
	}

	PinnedHostMemory& PinnedHostMemory::operator=(PinnedHostMemory&& other) noexcept {
		if (this != &other) {
			Release();

			m_mem = std::exchange(other.m_mem, nullptr);
			m_sizeBytes = std::exchange(other.m_sizeBytes, 0);
		}

		return *this;
	}

	void PinnedHostMemory::Allocate(std::size_t sizeBytes) {
		Release();

		if (sizeBytes == 0) {
			return;
		}

		cudaCheck(cudaMallocHost(&m_mem, sizeBytes));
		m_sizeBytes = sizeBytes;
	}

	void PinnedHostMemory::Release() noexcept {
		if (m_mem) {
			// Destructors should not throw.
			cudaFreeHost(m_mem);

			m_mem = nullptr;
			m_sizeBytes = 0;
		}
	}

	void* PinnedHostMemory::Mem() const {
		return m_mem;
	}

	std::size_t PinnedHostMemory::SizeBytes() const {
		return m_sizeBytes;
	}
}
