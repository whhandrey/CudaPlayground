#include "IMotionViewRenderer.h"
#include "../State/IMotionGpuPipelineState.h"
#include <Cuda/Context.h>
#include <Cuda/Motion/Visualization/Visualization.h>
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
		auto output_view = m_state.View(ViewType::ConfMap);
		cuda::motion::visualization::Conf(m_state.Stats(), output_view, m_ctx);
	}

	IMotionViewRenderer::Ptr IMotionViewRenderer::Create(
		cuda::KernelContext& ctx,
		state::IMotionGpuPipelineState& state,
		ViewType type)
	{
		switch (type)
		{
		case cuda::motion::render::ViewType::ConfMap:
			return std::make_unique<ConfidenceViewRenderer>(ctx, state);
		case cuda::motion::render::ViewType::VisualizationMap:
			return std::make_unique<ConfidenceViewRenderer>(ctx, state);
		}

		throw std::logic_error("IMotionViewRenderer::Create: invalid renderer type");
	}
}
