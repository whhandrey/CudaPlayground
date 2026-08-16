#pragma once
#include <cstddef>

namespace cuda::memory {
	class LinearDeviceMemory {
	public:
		LinearDeviceMemory(size_t sizeBytes);
		~LinearDeviceMemory();

		void* Mem() const;
		size_t SizeBytes() const;

	private:
		void* m_mem = nullptr;
		size_t m_sizeBytes;
	};
}
