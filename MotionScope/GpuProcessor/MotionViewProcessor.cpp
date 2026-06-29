#include "MotionViewProcessor.h"

namespace {
	bool EqVersion(const gpu::motion::FrameVersion& v1, const gpu::motion::VersionedFrame& v2) {
		return v1.index == v2.index && v1.generation == v2.generation;
	}
}

namespace gpu::motion {
	MotionViewProcessor::MotionViewProcessor(std::unique_ptr<cuda::motion::IMotionViewProcessor>&& proc)
		: m_processor{ std::move(proc) }
	{
	}

	std::map<ViewType, Image<image::vec4uc>> MotionViewProcessor::RenderViews(
		const VersionedFrame& prev,
		const VersionedFrame& curr,
		const std::vector<ViewType>& types)
	{
		const bool skipAnalysis = EqVersion(m_prev, prev) && EqVersion(m_curr, curr);
		if (!skipAnalysis) {
			m_processor->Analyze(prev.img, curr.img);

			m_prev = { prev.index, prev.generation };
			m_curr = { curr.index, curr.generation };
		}

		return m_processor->RenderViews(types);
	}
}
