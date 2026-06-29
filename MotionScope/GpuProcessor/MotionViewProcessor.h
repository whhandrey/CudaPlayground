#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include <GpuProcessor/IMotionViewProcessor.h>

namespace gpu::motion {
	using cuda::motion::render::ViewType;
	using image::Image;

	struct VersionedFrame {
		image::ImageView<image::vec4uc> img;
		size_t index = 0;
		size_t generation = 0;
	};

	struct FrameVersion {
		size_t index = 0;
		size_t generation = 0;
	};

	class MotionViewProcessor {
	public:
		MotionViewProcessor(std::unique_ptr<cuda::motion::IMotionViewProcessor>&& proc);

		std::map<ViewType, Image<image::vec4uc>> RenderViews(
			const VersionedFrame& prev,
			const VersionedFrame& curr,
			const std::vector<ViewType>& types);

	private:
		FrameVersion m_prev;
		FrameVersion m_curr;

		std::unique_ptr<cuda::motion::IMotionViewProcessor> m_processor;
	};
}
