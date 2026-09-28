#pragma once
#include <Image/ImageTypes.h>
#include "../CudaCheck.h"

namespace cuda::gpu_image {
	template<class T>
	struct GpuAllocation {
		T* ptr{};
		std::size_t rowPitchBytes{};
	};

	struct GpuAlloc2D {
		template<class T>
		static GpuAllocation<T> Create(image::vec2ui dim)
		{
			GpuAllocation<T> result;
			cudaCheck(cudaMallocPitch(&result.ptr, &result.rowPitchBytes, dim.x * sizeof(T), dim.y));

			return result;
		}
	};

	struct GpuAlloc3D {
		template<class T>
		static GpuAllocation<T> Create(image::vec3ui dim)
		{
			const cudaExtent extent = make_cudaExtent(dim.x * sizeof(T), dim.y, dim.z);

			cudaPitchedPtr pitched{};
			cudaCheck(cudaMalloc3D(&pitched, extent));

			return {
				static_cast<T*>(pitched.ptr),
				pitched.pitch
			};
		}
	};

	template <class T, class Vec, class Allocator>
	class GpuImageBase {
	public:
		GpuImageBase() = default;

		GpuImageBase(Vec dim)
			: m_dim(dim)
		{
			auto allocation = Allocator::template Create<T>(dim);

			m_ptr = allocation.ptr;
			m_pitch = allocation.rowPitchBytes;
		}

		GpuImageBase(const GpuImageBase&) = delete;
		GpuImageBase& operator=(const GpuImageBase&) = delete;

		GpuImageBase(GpuImageBase&& other) noexcept
			: m_dim(other.m_dim)
			, m_pitch(other.m_pitch)
			, m_ptr(other.m_ptr)
		{
			other.m_dim = {};
			other.m_pitch = 0;
			other.m_ptr = nullptr;
		}

		GpuImageBase& operator=(GpuImageBase&& other) noexcept {
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

		Vec Dim() const {
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

		~GpuImageBase() {
			Destroy();
		}

	private:
		void Destroy() {
			if (m_ptr) {
				cudaFree(m_ptr);
			}

			m_ptr = nullptr;
			m_pitch = 0;
			m_dim = {};
		}

	private:
		Vec m_dim = {};
		size_t m_pitch = 0;
		T* m_ptr = nullptr;
	};

	template <class T>
	using GpuImage = GpuImageBase<T, image::vec2ui, GpuAlloc2D>;

	template <class T>
	using GpuVolume = GpuImageBase<T, image::vec3ui, GpuAlloc3D>;
}
