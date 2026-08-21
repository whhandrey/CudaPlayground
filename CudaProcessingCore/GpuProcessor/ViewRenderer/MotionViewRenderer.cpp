#include "IMotionViewRenderer.h"
#include "../State/IMotionGpuPipelineState.h"
#include <Cuda/KernelContext.h>
#include <Cuda/Motion/Visualization/Visualization.h>
#include <Cuda/Motion/Visualization/ArrowsMap.h>
#include <stdexcept>

namespace cuda::motion::render {
	using StatsRenderFunc = void (*) (
		const GpuImageView<BlockMatchStats>&,
		GpuImageView<uchar4>&,
		cuda::KernelContext&,
		image::vec2i,
		image::vec2ui
	);

	class ConfidenceViewRenderer : public IMotionViewRenderer {
	public:
		ConfidenceViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state);

		void Render() override;

	private:
		cuda::KernelContext& m_ctx;
		state::IMotionGpuPipelineState& m_state;
	};

	class StatsViewRenderer : public IMotionViewRenderer {
	public:
		StatsViewRenderer(
			cuda::KernelContext& ctx,
			state::IMotionGpuPipelineState& state,
			image::vec2i search_halfsize,
			ViewType viewType,
			StatsRenderFunc renderFunc
		);

		void Render() override;

	private:
		cuda::KernelContext& m_ctx;
		state::IMotionGpuPipelineState& m_state;
		const image::vec2i m_search_halfsize;

		ViewType m_viewType;
		StatsRenderFunc m_renderFunc;
	};

	class ArrowsMapViewRenderer : public IMotionViewRenderer {
	public:
		ArrowsMapViewRenderer(
			cuda::KernelContext& ctx,
			state::IMotionGpuPipelineState& state,
			image::vec2ui macroBlockDim,
			int groupSize,
			float thickness);

		void Render() override;

	private:
		cuda::KernelContext& m_ctx;
		state::IMotionGpuPipelineState& m_state;
		image::vec2ui m_macroBlockDim;

		int m_groupSize;
		float m_thickness;
	};
}

namespace cuda::motion::render {
	StatsViewRenderer::StatsViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state, image::vec2i search_halfsize, ViewType viewType, StatsRenderFunc renderFunc)
		: m_state{ state }
		, m_ctx{ ctx }
		, m_search_halfsize{ search_halfsize }
		, m_viewType{ viewType }
		, m_renderFunc{ renderFunc }
	{
	}

	void StatsViewRenderer::Render() {
		auto output_view = m_state.View(m_viewType);
		m_renderFunc(m_state.Stats(), output_view, m_ctx, m_search_halfsize, { 8, 8 });
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

	ArrowsMapViewRenderer::ArrowsMapViewRenderer(cuda::KernelContext& ctx, state::IMotionGpuPipelineState& state, image::vec2ui macroBlockDim, int groupSize, float thickness)
		: m_state{ state }
		, m_ctx{ ctx }
		, m_macroBlockDim{ macroBlockDim }
		, m_groupSize{ groupSize }
		, m_thickness{ thickness }
	{
	}

	void ArrowsMapViewRenderer::Render() {
		auto output_view = m_state.View(ViewType::ArrowsMap);
		cuda::motion::visualization::ArrowsMap(m_state.Stats(), output_view, m_ctx, m_macroBlockDim, m_groupSize, m_thickness);
	}

	IMotionViewRenderer::Ptr IMotionViewRenderer::Create(cuda::KernelContext& ctx, const ViewRendererParams& params, ViewType type)
	{
		switch (type)
		{
		case cuda::motion::render::ViewType::ConfMap:
			return std::make_unique<ConfidenceViewRenderer>(ctx, params.state);
		case cuda::motion::render::ViewType::MotionMap:
			return std::make_unique<StatsViewRenderer>(ctx, params.state, params.search_halfsize, ViewType::MotionMap, cuda::motion::visualization::MotionMap);
		case cuda::motion::render::ViewType::MagnitudeMap:
			return std::make_unique<StatsViewRenderer>(ctx, params.state, params.search_halfsize, ViewType::MagnitudeMap, cuda::motion::visualization::MagMap);
		case cuda::motion::render::ViewType::ArrowsMap:
			return std::make_unique<ArrowsMapViewRenderer>(ctx, params.state, params.macroBlockDim, params.groupSize, params.thickness);
		}

		throw std::logic_error("IMotionViewRenderer::Create: invalid renderer type");
	}
}
