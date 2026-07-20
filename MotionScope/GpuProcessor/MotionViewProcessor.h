#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include <GpuProcessor/IMotionViewProcessor.h>
#include "../DisplayTypes/DisplayTypes.h"

namespace app::motion {
	using cuda::motion::render::ViewType;
	using cuda::motion::IMotionViewProcessor;

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
		MotionViewProcessor(IMotionViewProcessor::Ptr&& proc);

		std::map<ViewType, Image<image::vec4uc>> AnalyzeAndRenderViews(
			const VersionedFrame& prev,
			const VersionedFrame& curr,
			const std::vector<ViewType>& views);

		std::map<ViewType, Image<image::vec4uc>> RenderViews(const std::vector<ViewType>& views);

	private:
		std::unique_ptr<cuda::motion::IMotionViewProcessor> m_processor;

		FrameVersion m_prev;
		FrameVersion m_curr;
	};
}
