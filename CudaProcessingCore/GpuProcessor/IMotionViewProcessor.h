#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include "ViewRenderer/ViewType.h"
#include <memory>
#include <map>

namespace cuda::motion {
	using image::Image;
	using image::ImageView;

	template <class T>
	struct IndexedCpuFrame {
		int index;
		ImageView<T> img;
	};

	class IMotionViewProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionViewProcessor>;

		virtual ~IMotionViewProcessor() = default;

		virtual std::map<render::ViewType, Image<image::vec4uc>> RenderViews(
			const IndexedCpuFrame<image::vec4uc>& prev,
			const IndexedCpuFrame<image::vec4uc>& curr,
			const std::vector<render::ViewType>& types) = 0;

		static Ptr Create();
	};
}
