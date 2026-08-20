#pragma once
#include <map>
#include <Image/ImageTypes.h>
#include <Cuda/Image/ImageGPU.h>
#include "IMotionGpuPipelineState.h"

namespace cuda::motion::state {
	class MotionGpuPipelineState : public IMotionGpuPipelineState {
	public:
		const GpuImageView<BlockMatchStats> Stats() const override;
		GpuImageView<uchar4> View(render::ViewType type) override;

		void Resize(render::ViewType viewType, image::vec2ui view_dim);
		void SetStats(const GpuImageView<BlockMatchStats>& stats);

		void ClearView(render::ViewType type, cudaStream_t stream);

	private:
		GpuImageView<BlockMatchStats> m_stats;
		std::map<render::ViewType, cuda::gpu_image::ImageGPU<uchar4>> m_renderedViews;
	};
}
