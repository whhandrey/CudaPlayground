#pragma once
#include "ViewType.h"
#include <memory>
#include <Image/ImageTypes.h>

namespace cuda {
	struct KernelContext;

	namespace motion::state {
		class IMotionGpuPipelineState;
	}
}

namespace cuda::motion::render {
	struct ViewRendererParams {
		state::IMotionGpuPipelineState& state;
		image::vec2i search_halfsize;
		image::vec2ui macroBlockDim;
		int groupSize;
		float thickness;
	};

	class IMotionViewRenderer {
	public:
		using Ptr = std::unique_ptr<IMotionViewRenderer>;

		virtual ~IMotionViewRenderer() = default;
		virtual void Render() = 0;

		static Ptr Create(cuda::KernelContext& ctx, const ViewRendererParams& params, ViewType type);
	};
}
