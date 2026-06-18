#pragma once
#include <vector>
#include "ImageTypes.h"

namespace image {
	template <class T>
	class Image {
	public:
		Image(vec2ui dim)
			: m_dim(dim)
			, m_data(dim.x * dim.y)
		{
		}

		Image(unsigned char* ptr, vec2ui dim)
			: Image(dim)
		{
			memcpy(m_data.data(), ptr, dim.x * dim.y * sizeof(T));
		}

		Image(const Image&) = delete;
		Image& operator=(const Image&) = delete;

		Image(Image&&) noexcept = default;
		Image& operator=(Image&&) noexcept = default;

		vec2ui Dim() const {
			return m_dim;
		}

		const T* Data() const {
			return m_data.data();
		}

		T* Data() {
			return m_data.data();
		}

	private:
		vec2ui m_dim;
		std::vector<T> m_data;
	};
}
