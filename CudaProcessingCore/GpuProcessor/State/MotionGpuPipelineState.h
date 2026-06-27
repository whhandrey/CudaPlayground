#pragma once
#include <map>
#include <Image/ImageTypes.h>

#include "IMotionGpuPipelineState.h"
#include "../../DeviceImage/ImageGPU.h"

namespace cuda::motion::state {
	class MotionGpuPipelineState : public IMotionGpuPipelineState {
	public:
		const GpuImageView<BlockMatchStats> Stats() const override;
		GpuImageView<uchar4> View(render::ViewType type) override;

		void Resize(image::vec2ui img_dim);
		void SetStats(const GpuImageView<BlockMatchStats>& stats);

	private:
		GpuImageView<BlockMatchStats> m_stats;
		std::map<render::ViewType, ImageGPU<uchar4>> m_renderedViews;
	};
}
