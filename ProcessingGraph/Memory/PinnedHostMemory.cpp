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
		: m_data{ std::exchange(other.m_data, nullptr) }
		, m_sizeBytes{ std::exchange(other.m_sizeBytes, 0) }
	{
	}

	PinnedHostMemory& PinnedHostMemory::operator=(PinnedHostMemory&& other) noexcept {
		if (this != &other) {
			Release();

			m_data = std::exchange(other.m_data, nullptr);
			m_sizeBytes = std::exchange(other.m_sizeBytes, 0);
		}

		return *this;
	}

	void PinnedHostMemory::Allocate(std::size_t sizeBytes) {
		Release();

		if (sizeBytes == 0) {
			return;
		}

		cudaCheck(cudaMallocHost(&m_data, sizeBytes));
		m_sizeBytes = sizeBytes;
	}

	void PinnedHostMemory::Release() noexcept {
		if (m_data) {
			// Destructors should not throw.
			cudaFreeHost(m_data);

			m_data = nullptr;
			m_sizeBytes = 0;
		}
	}

	void* PinnedHostMemory::Data() const {
		return m_data;
	}

	std::size_t PinnedHostMemory::SizeBytes() const {
		return m_sizeBytes;
	}
}
