#pragma once
#include "ImageTypes.h"

namespace image {
	template <class T>
	struct GpuImageView {
		using SampleType = T;

		T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};

	template <class T>
	struct CpuImageView {
		using SampleType = T;

		const T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};

	template <class T>
	struct PinnedImageView {
		using SampleType = T;

		const T* m_ptr;
		image::vec2ui m_dim;
		size_t m_pitch;
	};
}
