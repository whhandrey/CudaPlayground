#pragma once
#include <memory>
#include <map>
#include <QImage>
#include <GpuProcessor/ViewRenderer/ViewType.h>
#include <Motion/Debug/Stats.h>
#include "../../DisplayTypes/DisplayTypes.h"

namespace app::motion {
	using cuda::motion::render::ViewType;
	using ::motion::debug::StatsPacket;

	class MotionViewProcessor;

	struct AnalyseInput {
		app::FramePair framePair;
		size_t generation;
		QImage prev;
		QImage curr;
		std::vector<SlottedView> requestedViews;
	};

	struct AnalyseResult {
		app::FramePair framePair;
		size_t generation;
		QImage prev;
		QImage curr;
		std::vector<SlottedImage> views;
		StatsPacket stats;
	};

	using AnalyseResultCb = std::function<void(AnalyseResult&&)>;

	struct RenderViewsInput {
		app::FramePair framePair;
		size_t generation;
		SlottedView requestedView;
	};

	struct RenderViewsResult {
		app::FramePair framePair;
		size_t generation;
		SlottedImage view;
		StatsPacket stats;
	};

	using RenderViewsResultCb = std::function<void(RenderViewsResult&&)>;

	class IGpuJob {
	public:
		using Ptr = std::unique_ptr<IGpuJob>;

		virtual ~IGpuJob() = default;
		virtual void Execute(MotionViewProcessor& proc) = 0;
	};

	IGpuJob::Ptr CreateAnalyzeAndRenderJob(const AnalyseInput& input, AnalyseResultCb&& callback);
	IGpuJob::Ptr CreateRenderViewsJob(const RenderViewsInput& input, RenderViewsResultCb&& callback);
}
