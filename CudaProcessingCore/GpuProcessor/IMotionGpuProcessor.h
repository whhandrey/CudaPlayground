#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>

namespace cuda::motion {
	using image::Image;
	using image::ImageView;

	enum class Type {
		ConfidenceMap,
		VisualizationMap
	};

	class IMotionGpuProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionGpuProcessor>;

		virtual ~IMotionGpuProcessor() = default;
		virtual void Process(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) = 0;

		static Ptr Create();
	};
}
