#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include <cuda_runtime.h>

namespace cuda_processing {
	using image::Image;
	using image::ImageView;

	class IMotionGpuProcessor {
	public:
		virtual ~IMotionGpuProcessor() = default;
		virtual Image<uchar4> Process(const ImageView<uchar4>& prev, const ImageView<uchar4>& curr) = 0;
	};
}
