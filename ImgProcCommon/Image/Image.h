#pragma once
#include <vector>
#include "ImageTypes.h"

namespace image {
	template <class T>
	struct CpuImage {
	public:
		CpuImage(vec2ui dim)
			: m_dim(dim)
			, m_data(dim.x * dim.y)
		{
		}

		CpuImage(unsigned char* ptr, vec2ui dim)
			: CpuImage(dim)
		{
			memcpy(m_data.data(), ptr, dim.x * dim.y * sizeof(T));
		}

		vec2ui Dim() const {
			return m_dim;
		}

		size_t Pitch() {
			return m_dim.x * sizeof(T);
		}

		const T* Data() const {
			return m_data.data();
		}

		T* Data() {
			return m_data.data();
		}

	public:
		vec2ui m_dim;
		std::vector<T> m_data;
	};

	template <class T>
	struct CpuVolume {
	public:
		CpuVolume(vec3ui dim)
			: m_dim(dim)
			, m_data(dim.x * dim.y * dim.z)
		{
		}

		CpuVolume(unsigned char* ptr, vec3ui dim)
			: CpuVolume(dim)
		{
			memcpy(m_data.data(), ptr, dim.x * dim.y * dim.z * sizeof(T));
		}

		vec3ui Dim() const {
			return m_dim;
		}

		size_t Pitch() {
			return m_dim.x * sizeof(T);
		}

		size_t SlicePitch() {
			return Pitch() * m_dim.y;
		}

		const T* Data() const {
			return m_data.data();
		}

		T* Data() {
			return m_data.data();
		}

	public:
		vec3ui m_dim;
		std::vector<T> m_data;
	};
}
