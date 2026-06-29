#include "IMotionViewProcessor.h"
#include "State/MotionGpuPipelineState.h"
#include "ViewRenderer/IMotionViewRenderer.h"
#include "../DeviceImage/GpuImageView.h"
#include "../DeviceImage/GpuImageTransfer.h"

#include <Cuda/MathUtils.h>

namespace {
	bool EqDim(image::vec2ui dim1, image::vec2ui dim2) {
		return dim1.x == dim2.x && dim1.y == dim2.y;
	}
	
	template <class T>
	bool Valid(const cuda::ImageGPU<T>& img) {
		return img.Dim().x != 0 && img.Dim().y != 0;
	}

	class MockProfiler : public cuda::profiler::IProfiler {
	public:
		void Profile(const std::string&, float) override {}
	};

	cuda::motion::BlockMatchingParams GetParams() {
		const image::vec2ui blockDim = { 8, 8 };
		const image::vec2ui macroBlockDim = { 16, 16 };
		const image::vec2i search_halfsize = { 3, 3 };

		return {
			blockDim,
			macroBlockDim,
			search_halfsize
		};
	}
}

namespace cuda::motion {
	class MotionGpuPipeline : public IMotionViewProcessor {
	public:
		MotionGpuPipeline();
		~MotionGpuPipeline();

	public:
		void Analyze(
			const image::ImageView<image::vec4uc>& prev,
			const image::ImageView<image::vec4uc>& curr) override;

		std::map<render::ViewType, image::Image<image::vec4uc>> RenderViews(
			const std::vector<render::ViewType>& types) override;

	private:
		void AllocMem(image::vec2ui dim);

	private:
		cuda::KernelContext m_ctx;
		state::MotionGpuPipelineState m_state;
		const BlockMatchingParams m_params;

		ImageGPU<uchar4> m_prev;
		ImageGPU<uchar4> m_curr;

		ImageGPU<BlockMatchStats> m_stats;
	};
}

namespace cuda::motion {
	MotionGpuPipeline::MotionGpuPipeline()
		: m_params{ GetParams() }
	{
		cudaStream_t stream;
		cudaCheck(cudaStreamCreate(&stream));

		m_ctx = { stream, std::make_unique<MockProfiler>() };
	}

	MotionGpuPipeline::~MotionGpuPipeline() {
		cudaCheck(cudaStreamDestroy(m_ctx.m_stream));
	}

	void MotionGpuPipeline::Analyze(const image::ImageView<image::vec4uc>& prev,
		const image::ImageView<image::vec4uc>& curr)
	{
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuPipeline::Analyze: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);

		cuda::gpu_image::UploadCompatible(prev, m_prev, m_ctx.m_stream);
		cuda::gpu_image::UploadCompatible(curr, m_curr, m_ctx.m_stream);

		auto outView = cuda::gpu_image::MakeImageView(m_stats);
		m_state.SetStats(outView);

		cuda::motion::BlockMatching(
			cuda::gpu_image::MakeImageView(m_prev),
			cuda::gpu_image::MakeImageView(m_curr),
			outView,
			m_params,
			m_ctx
		);
	}

	std::map<render::ViewType, image::Image<image::vec4uc>> MotionGpuPipeline::RenderViews(
		const std::vector<render::ViewType>& types)
	{
		if (!Valid(m_stats)) {
			throw std::logic_error("MotionGpuPipeline::RenderViews: analysis has not been performed");
		}

		std::map<render::ViewType, image::Image<image::vec4uc>> output;

		for (const auto type : types) {
			auto renderer = render::IMotionViewRenderer::Create(m_ctx, m_state, type);
			renderer->Render();

			output.emplace(type, cuda::gpu_image::DownloadCompatible<image::vec4uc>(m_state.View(type), m_ctx.m_stream));
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
		m_state.Resize(motion_dim);
	}

	IMotionViewProcessor::Ptr IMotionViewProcessor::Create() {
		// not really pipeline yet but let's see how it goes
		return std::make_unique<MotionGpuPipeline>();
	}
}
