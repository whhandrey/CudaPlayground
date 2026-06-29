#pragma once
#include "ViewType.h"
#include <memory>

namespace cuda {
	struct KernelContext;

	namespace motion::state {
		class IMotionGpuPipelineState;
	}
}

namespace cuda::motion::render {
	class IMotionViewRenderer {
	public:
		using Ptr = std::unique_ptr<IMotionViewRenderer>;

		virtual ~IMotionViewRenderer() = default;
		virtual void Render() = 0;

		static Ptr Create(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state, ViewType type);
	};
}
