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
		cuda::KernelContext& ctx;
		state::IMotionGpuPipelineState& state;
		image::vec2i search_halfsize;
	};

	class IMotionViewRenderer {
	public:
		using Ptr = std::unique_ptr<IMotionViewRenderer>;

		virtual ~IMotionViewRenderer() = default;
		virtual void Render() = 0;

		static Ptr Create(const ViewRendererParams& params, ViewType type);
	};
}
