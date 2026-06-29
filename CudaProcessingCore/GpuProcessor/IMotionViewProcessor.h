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
	struct VersionedCpuFrame {
		ImageView<T> img;
		size_t index = 0;
		size_t generation = 0;
	};

	class IMotionViewProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionViewProcessor>;

		virtual ~IMotionViewProcessor() = default;

		virtual std::map<render::ViewType, Image<image::vec4uc>> RenderViews(
			const VersionedCpuFrame<image::vec4uc>& prev,
			const VersionedCpuFrame<image::vec4uc>& curr,
			const std::vector<render::ViewType>& types) = 0;

		static Ptr Create();
	};
}
