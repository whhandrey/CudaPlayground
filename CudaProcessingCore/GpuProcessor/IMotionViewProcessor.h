#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include <Motion/Debug/Stats.h>
#include "ViewRenderer/ViewType.h"
#include <memory>
#include <map>

namespace cuda::motion {
	using ::motion::debug::StatsPacket;

	class IMotionViewProcessor {
	public:
		using Ptr = std::unique_ptr<IMotionViewProcessor>;

		virtual ~IMotionViewProcessor() = default;

		virtual void Analyze(
			const image::CpuImageView<const image::vec4uc>& prev,
			const image::CpuImageView<const image::vec4uc>& curr) = 0;

		virtual std::map<render::ViewType, image::Image<image::vec4uc>> RenderViews(
			const std::vector<render::ViewType>& types) = 0;

		virtual StatsPacket TakeLastStats() = 0;

		static Ptr Create();
	};
}
