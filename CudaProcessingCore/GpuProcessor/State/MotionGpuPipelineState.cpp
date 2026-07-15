#include "MotionGpuPipelineState.h"
#include "../ViewRenderer/ViewType.h"
#include "../../DeviceImage/GpuImageView.h"
#include <array>

namespace cuda::motion::state {
	using cuda::motion::render::ViewType;

	const GpuImageView<BlockMatchStats> MotionGpuPipelineState::Stats() const {
		return m_stats;
	}

	GpuImageView<uchar4> MotionGpuPipelineState::View(render::ViewType type) {
		if (m_renderedViews.find(type) == m_renderedViews.end()) {
			throw std::logic_error("MotionGpuPipelineState::View: not supported type");
		}

		return cuda::gpu_image::MakeImageView(m_renderedViews.at(type));
	}

	void MotionGpuPipelineState::Resize(image::vec2ui img_dim) {
		const std::array<ViewType, 3> allViews {
			ViewType::ConfMap,
			ViewType::MotionMap,
			ViewType::MagnitudeMap
		};

		for (const auto view : allViews) {
			m_renderedViews[view] = ImageGPU<uchar4>(img_dim);
		}
	}

	void MotionGpuPipelineState::SetStats(const GpuImageView<BlockMatchStats>& stats) {
		m_stats = stats;
	}
}
