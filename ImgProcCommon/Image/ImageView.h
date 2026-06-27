#pragma once
#include "ImageTypes.h"

namespace image {
	template <class T>
	struct GpuImageView {
		const T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};

	template <class T>
	struct ImageView {
		const T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};
}
