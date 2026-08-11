#include "IMotionViewProcessor.h"
#include "State/MotionGpuPipelineState.h"
#include "ViewRenderer/IMotionViewRenderer.h"
#include "../DeviceImage/GpuImageView.h"
#include "../DeviceImage/GpuImageTransfer.h"
#include "../Motion/Debug/Collector.h"

#include <Cuda/MathUtils.h>
#include <Cuda/TimedCudaCall.h>
#include <Profiler/Profiler.h>

namespace {
	using cuda::motion::render::ViewType;

	bool EqDim(image::vec2ui dim1, image::vec2ui dim2) {
		return dim1.x == dim2.x && dim1.y == dim2.y;
	}
	
	template <class T>
	bool Valid(const cuda::ImageGPU<T>& img) {
		return img.Dim().x != 0 && img.Dim().y != 0;
	}

	cuda::motion::BlockMatchingParams GetDefaultParams() {
		const image::vec2ui blockDim = { 8, 8 };
		const image::vec2ui macroBlockDim = { 16, 16 };
		const image::vec2i search_halfsize = { 3, 3 };

		return {
			blockDim,
			macroBlockDim,
			search_halfsize
		};
	}

	std::map<ViewType, image::vec2ui> RenderViewsDims(image::vec2ui motion_dim, image::vec2ui arrows_dim) {
		return {
			{ ViewType::ConfMap, motion_dim },
			{ ViewType::MotionMap, motion_dim },
			{ ViewType::MagnitudeMap, motion_dim },
			{ ViewType::ArrowsMap, arrows_dim }
		};
	}
}

namespace cuda::motion {
	class MotionGpuPipeline : public IMotionViewProcessor {
	public:
		MotionGpuPipeline(std::unique_ptr<debug::Collector> collector);
		~MotionGpuPipeline();

	public:
		void Analyze(
			const image::CpuImageView<image::vec4uc>& prev,
			const image::CpuImageView<image::vec4uc>& curr) override;

		std::map<render::ViewType, image::Image<image::vec4uc>> RenderViews(
			const std::vector<render::ViewType>& views) override;

		StatsPacket TakeLastStats() override;

	private:
		void AllocMem(image::vec2ui dim);
		void WriteGpuStats(const std::string& scope, const std::string& group);

	private:
		cuda::KernelContext m_ctx;
		state::MotionGpuPipelineState m_state;

		std::unique_ptr<profile::SimpleProfiler> m_gpuProfiler;
		std::unique_ptr<debug::Collector> m_collector;
		
		BlockMatchingParams m_params;

		ImageGPU<uchar4> m_prev;
		ImageGPU<uchar4> m_curr;

		ImageGPU<BlockMatchStats> m_stats;
	};
}

namespace cuda::motion {
	MotionGpuPipeline::MotionGpuPipeline(std::unique_ptr<debug::Collector> collector)
		: m_params{ GetDefaultParams() }
		, m_collector{ std::move(collector) }
		, m_gpuProfiler{ std::make_unique<profile::SimpleProfiler>() }
	{
		cudaStream_t stream;
		cudaCheck(cudaStreamCreate(&stream));

		m_ctx = { stream, m_gpuProfiler.get() };
	}

	MotionGpuPipeline::~MotionGpuPipeline() {
		cudaCheck(cudaStreamDestroy(m_ctx.m_stream));
	}

	void MotionGpuPipeline::Analyze(const image::CpuImageView<image::vec4uc>& prev,
		const image::CpuImageView<image::vec4uc>& curr)
	{
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuPipeline::Analyze: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);

		cuda::TimedCall("UploadCompatible(prev)", m_ctx, [this, &prev]() {
			cuda::gpu_image::UploadCompatible(prev, m_prev, m_ctx.m_stream);
		});

		cuda::TimedCall("UploadCompatible(curr)", m_ctx, [this, &curr]() {
			cuda::gpu_image::UploadCompatible(curr, m_curr, m_ctx.m_stream);
		});

		auto outView = cuda::gpu_image::MakeImageView(m_stats);
		m_state.SetStats(outView);

		cuda::motion::BlockMatching(
			cuda::gpu_image::MakeImageView(m_prev),
			cuda::gpu_image::MakeImageView(m_curr),
			outView,
			m_params,
			m_ctx
		);

		m_collector->AddParams(m_params);
		WriteGpuStats("Algo", "GpuStats");
	}

	std::map<render::ViewType, image::Image<image::vec4uc>> MotionGpuPipeline::RenderViews(
		const std::vector<render::ViewType>& views)
	{
		if (!Valid(m_stats)) {
			throw std::logic_error("MotionGpuPipeline::RenderViews: analysis has not been performed");
		}

		const int groupSize = 4;
		const float thickness = 4.0f;

		const auto params = render::ViewRendererParams {
			m_ctx,
			m_state,
			m_params.search_halfsize,
			m_params.macroBlockDim,
			groupSize,
			thickness
		};

		std::map<render::ViewType, image::Image<image::vec4uc>> output;

		int viewIndex = 0;
		for (const auto view : views) {
			m_state.ClearView(view, m_ctx.m_stream);

			auto renderer = render::IMotionViewRenderer::Create(params, view);
			renderer->Render();

			output.emplace(view, cuda::gpu_image::DownloadCompatible<image::vec4uc>(m_state.View(view), m_ctx.m_stream));
			WriteGpuStats("RenderView" + std::to_string(viewIndex++), "GpuStats");
		}

		cudaCheck(cudaStreamSynchronize(m_ctx.m_stream));
		return output;
	}

	void MotionGpuPipeline::AllocMem(image::vec2ui dim) {
		const auto curr_dim = m_prev.Dim();

		if (curr_dim.x == dim.x && curr_dim.y == dim.y) {
			return;
		}

		m_prev = ImageGPU<uchar4>(dim);
		m_curr = ImageGPU<uchar4>(dim);

		const auto motion_dim = cuda::motion::MotionOutputDim(dim, m_params.macroBlockDim);
		m_stats = ImageGPU<BlockMatchStats>(motion_dim);

		const auto arrows_dim = dim;
		const auto viewsDims = RenderViewsDims(motion_dim, arrows_dim);

		for (const auto viewDim : viewsDims) {
			m_state.Resize(viewDim.first, viewDim.second);
		}
	}

	void MotionGpuPipeline::WriteGpuStats(const std::string& scope, const std::string& group)
	{
		for (const auto& [name, time] : m_gpuProfiler->GetRuns()) {
			m_collector->AddStat(scope, group, name, time);
		}

		m_gpuProfiler->Clear();
	}

	StatsPacket MotionGpuPipeline::TakeLastStats()
	{
		return m_collector->TakeStats();
	}

	IMotionViewProcessor::Ptr IMotionViewProcessor::Create() {
		// not really pipeline yet but let's see how it goes
		return std::make_unique<MotionGpuPipeline>(std::make_unique<debug::Collector>());
	}
}
