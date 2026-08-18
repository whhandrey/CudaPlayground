#pragma once
#include <cstddef>

namespace cuda::memory {
	class LinearDeviceMemory {
	public:
		LinearDeviceMemory() = default;
		~LinearDeviceMemory();

		LinearDeviceMemory(const LinearDeviceMemory&) = delete;
		LinearDeviceMemory& operator=(const LinearDeviceMemory&) = delete;

		LinearDeviceMemory(LinearDeviceMemory&& other) noexcept;
		LinearDeviceMemory& operator=(LinearDeviceMemory&& other) noexcept;

		void Allocate(size_t sizeBytes);
		void Release();

		void* Mem() const;
		size_t SizeBytes() const;

	private:
		void* m_mem = nullptr;
		size_t m_sizeBytes = 0;
	};
}
