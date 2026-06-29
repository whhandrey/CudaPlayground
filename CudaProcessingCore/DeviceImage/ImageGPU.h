#pragma once
#include <Image/ImageTypes.h>
#include <Cuda/CudaCheck.h>

namespace cuda {
	template <class T>
	class ImageGPU {
	public:
		ImageGPU() = default;

		ImageGPU(image::vec2ui dim)
			: m_dim(dim)
		{
			Create();
		}

		ImageGPU(const ImageGPU&) = delete;
		ImageGPU& operator=(const ImageGPU&) = delete;

		ImageGPU(ImageGPU&& other) noexcept
			: m_dim(other.m_dim)
			, m_pitch(other.m_pitch)
			, m_ptr(other.m_ptr)
		{
			other.m_dim = {};
			other.m_pitch = 0;
			other.m_ptr = nullptr;
		}

		ImageGPU& operator=(ImageGPU&& other) noexcept {
			if (this != &other) {
				Destroy();

				m_dim = other.m_dim;
				m_pitch = other.m_pitch;
				m_ptr = other.m_ptr;

				other.m_dim = {};
				other.m_pitch = 0;
				other.m_ptr = nullptr;
			}

			return *this;
		}

		image::vec2ui Dim() const {
			return m_dim;
		}

		size_t Pitch() const {
			return m_pitch;
		}

		const T* Data() const {
			return m_ptr;
		}

		T* Data() {
			return m_ptr;
		}

		~ImageGPU() {
			Destroy();
		}

	private:
		void Create() {
			cudaCheck(cudaMallocPitch(&m_ptr, &m_pitch, m_dim.x * sizeof(T), m_dim.y));
		}

		void Destroy() {
			if (m_ptr) {
				cudaFree(m_ptr);
				m_ptr = nullptr;
			}

			m_pitch = 0;
			m_dim = {};
		}

	private:
		image::vec2ui m_dim = {};
		size_t m_pitch = 0;
		T* m_ptr = nullptr;
	};
}
