#include "IMotionViewRenderer.h"
#include "../State/IMotionGpuPipelineState.h"
#include <Cuda/Context.h>
#include <stdexcept>

namespace cuda::motion::render {
	class ConfidenceViewRenderer : public IMotionViewRenderer {
	public:
		ConfidenceViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state);

		void Render() override;

	private:
		cuda::KernelContext& m_ctx;
		state::IMotionGpuPipelineState& m_state;
	};
}

namespace cuda::motion::render {
	ConfidenceViewRenderer::ConfidenceViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state)
		: m_state{ state }
		, m_ctx{ ctx }
	{
	}

	void ConfidenceViewRenderer::Render() {

	}

	IMotionViewRenderer::Ptr IMotionViewRenderer::Create(ViewType type) {
		switch (type)
		{
		case cuda::motion::render::ViewType::ConfidenceMap:
			break;
		case cuda::motion::render::ViewType::VisualizationMap:
			break;
		}

		throw std::logic_error("IMotionViewRenderer::Create: invalid renderer type");
	}
}
