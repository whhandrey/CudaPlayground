#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include <memory>

namespace cuda::motion {
	using image::Image;
	using image::ImageView;

	namespace render {
		enum class ViewType;
	}

	class IMotionViewProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionViewProcessor>;
		using ImageCpuU4 = Image<image::vec4uc>;

		virtual ~IMotionViewProcessor() = default;

		virtual std::vector<ImageCpuU4> RenderViews(
			const ImageView<image::vec4uc>& prev,
			const ImageView<image::vec4uc>& curr,
			const std::vector<render::ViewType> types) = 0;

		static Ptr Create();
	};
}
