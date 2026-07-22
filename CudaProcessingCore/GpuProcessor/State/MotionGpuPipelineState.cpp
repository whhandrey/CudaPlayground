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

	void MotionGpuPipelineState::Resize(render::ViewType viewType, image::vec2ui view_dim) {
		m_renderedViews[viewType] = ImageGPU<uchar4>(view_dim);
	}

	void MotionGpuPipelineState::SetStats(const GpuImageView<BlockMatchStats>& stats) {
		m_stats = stats;
	}

	void MotionGpuPipelineState::ClearView(render::ViewType type, cudaStream_t stream) {
		if (m_renderedViews.find(type) == m_renderedViews.end()) {
			throw std::logic_error("MotionGpuPipelineState::ClearView: not supported type");
		}

		auto& img = m_renderedViews.at(type);
		cudaCheck(cudaMemset2DAsync(img.Data(), img.Pitch(), 0, img.Dim().x * sizeof(uchar4), img.Dim().y, stream));
	}
}
