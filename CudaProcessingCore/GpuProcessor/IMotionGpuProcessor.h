#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>

namespace cuda {
	using image::Image;
	using image::ImageView;

	class IMotionGpuProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionGpuProcessor>;

		virtual ~IMotionGpuProcessor() = default;
		virtual Image<unsigned char> Process(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) = 0;

		static Ptr Create();
	};
}
