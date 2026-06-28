#include "MotionGpuPipelineState.h"
#include "../../DeviceImage/GpuImageView.h"

namespace cuda::motion::state {
	const GpuImageView<BlockMatchStats> MotionGpuPipelineState::Stats() const {
		return m_stats;
	}

	GpuImageView<uchar4> MotionGpuPipelineState::View(render::ViewType type) {
		if (m_renderedViews.find(type) == m_renderedViews.end()) {
			throw std::logic_error("MotionGpuPipelineState::View: not supported type");
		}

		return image_view::MakeImageView(m_renderedViews.at(type));
	}

	void MotionGpuPipelineState::Resize(image::vec2ui img_dim) {

	}

	void MotionGpuPipelineState::SetStats(const GpuImageView<BlockMatchStats>& stats) {
		m_stats = stats;
	}
}
