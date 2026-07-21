#include "MotionViewProcessor.h"
#include <algorithm>

namespace {
	bool EqVersion(const app::motion::FrameVersion& v1, const app::motion::VersionedFrame& v2) {
		return v1.index == v2.index && v1.generation == v2.generation;
	}
}

namespace app::motion {
	MotionViewProcessor::MotionViewProcessor(IMotionViewProcessor::Ptr&& proc)
		: m_processor{ std::move(proc) }
	{
	}

	std::map<ViewType, Image<image::vec4uc>> MotionViewProcessor::AnalyzeAndRenderViews(
		const VersionedFrame& prev,
		const VersionedFrame& curr,
		const std::vector<ViewType>& views)
	{
		const bool skipAnalysis = EqVersion(m_prev, prev) && EqVersion(m_curr, curr);
		if (!skipAnalysis) {
			m_processor->Analyze(prev.img, curr.img);

			m_prev = { prev.index, prev.generation };
			m_curr = { curr.index, curr.generation };
		}

		return RenderViews(views);
	}

	std::map<ViewType, Image<image::vec4uc>> MotionViewProcessor::RenderViews(const std::vector<ViewType>& views)
	{
		if (m_prev.index == m_curr.index) {
			throw std::logic_error("MotionViewProcessor::RenderViews: analysis has not been run");
		}

		return m_processor->RenderViews(views);
	}

	StatsPacket MotionViewProcessor::TakeLastStats()
	{
		return m_processor->TakeLastStats();
	}
}
