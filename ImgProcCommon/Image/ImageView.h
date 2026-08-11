#pragma once
#include "ImageTypes.h"

namespace image {
	template <class T>
	struct GpuImageView {
		T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};

	template <class T>
	struct CpuImageView {
		const T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};
}
