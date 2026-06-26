#pragma once
#include <Image/ImageView.h>
#include <memory>

namespace cuda::motion {
	using image::ImageView;

	class IMotionStatsCollector {
	public:
		using Ptr = std::unique_ptr<IMotionStatsCollector>;

		virtual ~IMotionStatsCollector() = default;
		virtual void Collect(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) = 0;

		static Ptr CreateCollector();
	};
}
