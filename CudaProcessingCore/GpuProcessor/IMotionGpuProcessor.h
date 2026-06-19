#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>

namespace cuda_processing {
	using image::vec4uc;
	using image::Image;
	using image::ImageView;

	class IMotionGpuProcessor {
	public:
		virtual ~IMotionGpuProcessor() = default;
		virtual Image<vec4uc> Process(const ImageView<vec4uc>& prev, const ImageView<vec4uc>& curr) = 0;
	};
}
