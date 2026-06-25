#include "MotionGpuProcessor.h"
#include <Cuda/MathUtils.h>
#include "../DeviceImage/ImageTransfer.h"

namespace {
	bool EqDim(image::vec2ui dim1, image::vec2ui dim2) {
		return dim1.x == dim2.x && dim1.y == dim2.y;
	}

	template <class T>
	image::GpuImageView<T> MakeImageView(ImageGPU<T>& img) {
		return {
			img.Data(),
			img.Dim(),
			img.Pitch()
		};
	}

	template <class T>
	image::GpuImageView<T> MakeEmptyImageView() {
		return {
			nullptr,
			{},
			0
		};
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

namespace cuda {
	MotionGpuProcessor::MotionGpuProcessor()
		: m_params{ GetParams() }
	{
		cudaStream_t stream;
		cudaCheck(cudaStreamCreate(&stream));

		m_ctx = { stream, std::make_unique<MockProfiler>() };
	}

	MotionGpuProcessor::~MotionGpuProcessor() {
		cudaCheck(cudaStreamDestroy(m_ctx.m_stream));
	}

	Image<unsigned char> MotionGpuProcessor::Process(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) {
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuProcessor::Process: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);

		m_prev.UploadCompatible(prev, m_ctx.m_stream);
		m_curr.UploadCompatible(curr, m_ctx.m_stream);

		auto outView = MakeImageView(m_conf);
		auto dxdyView = MakeEmptyImageView<int2>();

		cuda::motion::BlockMatchingSimple(
			MakeImageView(m_prev),
			MakeImageView(m_curr),
			outView,
			dxdyView,
			m_params,
			m_ctx
		);

		auto image_out = image::ImageGpuToCpu(m_conf, m_ctx.m_stream);
		cudaStreamSynchronize(m_ctx.m_stream);

		return image_out;
	}

	void MotionGpuProcessor::AllocMem(image::vec2ui dim) {
		const auto curr_dim = m_prev.Dim();

		if (curr_dim.x == dim.x && curr_dim.y == dim.y) {
			return;
		}

		m_prev = ImageGPU<uchar4>(dim);
		m_curr = ImageGPU<uchar4>(dim);

		auto output_dim = cuda::math::Div(dim, m_params.macroBlockDim);
		m_conf = ImageGPU<unsigned char>(image::vec2ui{ output_dim.x, output_dim.y });
	}

	IMotionGpuProcessor::Ptr Create() {
		return std::make_unique<MotionGpuProcessor>();
	}
}
