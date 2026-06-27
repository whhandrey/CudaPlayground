#pragma once
#include <Cuda/Motion/BlockMatching.h>

namespace cuda::motion::render {
	enum class ViewType;
}

namespace cuda::motion::state {
	class IMotionGpuPipelineState {
	public:
		virtual ~IMotionGpuPipelineState() = default;

		virtual const GpuImageView<BlockMatchStats> Stats() const = 0;
		//virtual GpuImageView<BlockMatchStats>& Stats() = 0;

		virtual GpuImageView<uchar4> View(render::ViewType type) = 0;
	};
}
