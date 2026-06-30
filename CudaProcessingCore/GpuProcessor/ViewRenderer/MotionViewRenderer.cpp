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

	class MotionMapViewRenderer : public IMotionViewRenderer {
	public:
		MotionMapViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state, image::vec2i search_halfsize);

		void Render() override;

	private:
		cuda::KernelContext& m_ctx;
		state::IMotionGpuPipelineState& m_state;
		const image::vec2i m_search_halfsize;
	};
}

namespace cuda::motion::render {
	MotionMapViewRenderer::MotionMapViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state, image::vec2i search_halfsize)
		: m_state{ state }
		, m_ctx{ ctx }
		, m_search_halfsize{ search_halfsize }
	{
	}

	void MotionMapViewRenderer::Render() {
		auto output_view = m_state.View(ViewType::MotionMap);
		cuda::motion::visualization::MotionMap(m_state.Stats(), output_view, m_ctx, m_search_halfsize);
	}

	ConfidenceViewRenderer::ConfidenceViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state)
		: m_state{ state }
		, m_ctx{ ctx }
	{
	}

	void ConfidenceViewRenderer::Render() {
		auto output_view = m_state.View(ViewType::ConfMap);
		cuda::motion::visualization::Conf(m_state.Stats(), output_view, m_ctx);
	}

	IMotionViewRenderer::Ptr IMotionViewRenderer::Create(const ViewRendererParams& params, ViewType type)
	{
		switch (type)
		{
		case cuda::motion::render::ViewType::ConfMap:
			return std::make_unique<ConfidenceViewRenderer>(params.ctx, params.state);
		case cuda::motion::render::ViewType::MotionMap:
			return std::make_unique<MotionMapViewRenderer>(params.ctx, params.state, params.search_halfsize);
		}

		throw std::logic_error("IMotionViewRenderer::Create: invalid renderer type");
	}
}
