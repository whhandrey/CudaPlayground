#include "IMotionViewProcessor.h"
#include "State/MotionGpuPipelineState.h"
#include "ViewRenderer/IMotionViewRenderer.h"
#include "GpuImageView.h"

#include <Cuda/MathUtils.h>
#include "../DeviceImage/ImageTransfer.h"

namespace {
	bool EqDim(image::vec2ui dim1, image::vec2ui dim2) {
		return dim1.x == dim2.x && dim1.y == dim2.y;
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
		std::map<render::ViewType, ImageCpuU4> RenderViews(
			const ImageView<image::vec4uc>& prev,
			const ImageView<image::vec4uc>& curr,
			const std::vector<render::ViewType>& types) override;

	private:
		void AllocMem(image::vec2ui dim);
		void Analyze(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr);

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

	void MotionGpuPipeline::Analyze(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) {
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuPipeline::Analyze: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);

		m_prev.UploadCompatible(prev, m_ctx.m_stream);
		m_curr.UploadCompatible(curr, m_ctx.m_stream);

		auto outView = image_view::MakeImageView(m_stats);

		cuda::motion::BlockMatching(
			image_view::MakeImageView(m_prev),
			image_view::MakeImageView(m_curr),
			outView,
			m_params,
			m_ctx
		);
	}

	std::map<render::ViewType, Image<image::vec4uc>> MotionGpuPipeline::RenderViews(
		const ImageView<image::vec4uc>& prev,
		const ImageView<image::vec4uc>& curr,
		const std::vector<render::ViewType>& types)
	{
		Analyze(prev, curr);

		return {};
	}

	void MotionGpuPipeline::AllocMem(image::vec2ui dim) {
		const auto curr_dim = m_prev.Dim();

		if (curr_dim.x == dim.x && curr_dim.y == dim.y) {
			return;
		}

		m_prev = ImageGPU<uchar4>(dim);
		m_curr = ImageGPU<uchar4>(dim);

		m_stats = ImageGPU<BlockMatchStats>(cuda::motion::MotionOutputDim(dim, m_params.macroBlockDim));
	}

	IMotionViewProcessor::Ptr IMotionViewProcessor::Create() {
		return std::make_unique<MotionGpuPipeline>();
	}
}
