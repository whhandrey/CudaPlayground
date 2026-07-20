#pragma once
#include <memory>
#include <map>
#include <QImage>
#include <GpuProcessor/ViewRenderer/ViewType.h>
#include "../../DisplayTypes/DisplayTypes.h"

namespace app::motion {
	using cuda::motion::render::ViewType;

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
		std::map<ViewType, QImage> views;
	};

	using AnalyseResultCb = std::function<void(AnalyseResult&&)>;

	struct RenderViewsInput {
		app::FramePair framePair;
		size_t generation;
		std::vector<SlottedView> requestedViews;
	};

	struct RenderViewsResult {
		app::FramePair framePair;
		size_t generation;
		std::vector<SlottedImage> views;
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
