#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include "ViewRenderer/ViewType.h"
#include <memory>
#include <map>

namespace cuda::motion {
	using image::Image;
	using image::ImageView;

	struct IndexedFrame {
		int index;
		ImageView<const image::vec4uc> img;
	};

	class IMotionViewProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionViewProcessor>;
		using ImageCpuU4 = Image<image::vec4uc>;

		virtual ~IMotionViewProcessor() = default;

		virtual std::map<render::ViewType, ImageCpuU4> RenderViews(
			const IndexedFrame& prev,
			const IndexedFrame& curr,
			const std::vector<render::ViewType>& types) = 0;

		static Ptr Create();
	};
}
