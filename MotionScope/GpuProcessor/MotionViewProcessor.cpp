#include "MotionViewProcessor.h"

namespace {
	bool EqVersion(const app::motion::FrameVersion& v1, const app::motion::VersionedFrame& v2) {
		return v1.index == v2.index && v1.generation == v2.generation;
	}
}

namespace app::motion {
	MotionViewProcessor::MotionViewProcessor(std::unique_ptr<cuda::motion::IMotionViewProcessor>&& proc)
		: m_processor{ std::move(proc) }
	{
	}

	std::map<ViewType, Image<image::vec4uc>> MotionViewProcessor::AnalyzeAndRenderViews(
		const VersionedFrame& prev,
		const VersionedFrame& curr,
		const std::vector<SlottedView>& types)
	{
		const bool skipAnalysis = EqVersion(m_prev, prev) && EqVersion(m_curr, curr);
		if (!skipAnalysis) {
			m_processor->Analyze(prev.img, curr.img);

			m_prev = { prev.index, prev.generation };
			m_curr = { curr.index, curr.generation };
		}

		return RenderViews(types);
	}

	std::map<ViewType, Image<image::vec4uc>> MotionViewProcessor::RenderViews(const std::vector<SlottedView>& views)
	{
		if (m_prev.index == m_curr.index) {
			throw std::logic_error("MotionViewProcessor::RenderViews: analysis has not been run");
		}

		std::vector<ViewType> request;
		std::transform(views.begin(), views.end(), std::back_inserter(request), [](const auto& slottedView) {
			return slottedView.view;
		});

		m_cachedViews = views;

		return m_processor->RenderViews(request);
	}

	void MotionViewProcessor::OnDebugStats(StatsPacket&& stats)
	{
	}
}
