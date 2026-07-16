#pragma once
#include <memory>
#include <map>
#include <QImage>
#include <GpuProcessor/ViewRenderer/ViewType.h>
#include "../../DisplayTypes/DisplayTypes.h"

namespace gpu::motion {
	using cuda::motion::render::ViewType;

	class MotionViewProcessor;

	struct AnalyseInput {
		int frameIndex;
		size_t generation;
		QImage prev;
		QImage curr;
		std::vector<ViewType> requestedViews;
	};

	struct AnalyseResult {
		int frameIndex;
		size_t generation;
		QImage prev;
		QImage curr;
		std::map<ViewType, QImage> views;
	};

	using AnalyseResultCb = std::function<void(AnalyseResult&&)>;

	struct RequestedView {
		app::ViewSlot slot;
		ViewType view;
	};

	struct RenderViewsInput {
		int frameIndex;
		size_t generation;
		std::vector<RequestedView> requestedViews;
	};

	struct RenderViewsResult {
		int frameIndex;
		size_t generation;
		std::map<app::ViewSlot, QImage> views;
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
