#include "MotionGpuProcessor.h"
#include <Cuda/Motion.h>
#include <Cuda/MathUtils.h>

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

	class MockProfiler : public cuda::profiler::IProfiler {
	public:
		void Profile(const std::string&, float) override {}
	};
}

namespace cuda_processing {
	MotionGpuProcessor::MotionGpuProcessor(cudaStream_t stream)
		: m_stream{ stream }
	{
	}

	Image<uchar4> MotionGpuProcessor::Process(const ImageView<uchar4>& prev, const ImageView<uchar4>& curr) {
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuProcessor::Process: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);

		m_prev.Upload(prev, m_stream);
		m_curr.Upload(curr, m_stream);

		auto outView = MakeImageView(m_vecAndConf);
		auto p = cuda::motion::BlockMatchingParams {
			m_blockDim,
			m_macroBlockDim,
			m_search_halfsize
		};

		auto ctx = cuda::KernelContext {
			m_stream,
			std::make_unique<MockProfiler>()
		};

		cuda::motion::BlockMatchingSimple(
			MakeImageView(m_prev),
			MakeImageView(m_curr),
			outView,
			p,
			ctx
		);

		cudaStreamSynchronize(m_stream);
	}

	void MotionGpuProcessor::AllocMem(image::vec2ui dim) {
		const auto curr_dim = m_prev.Dim();

		if (curr_dim.x == dim.x && curr_dim.y == dim.y) {
			return;
		}

		m_prev = ImageGPU<uchar4>(dim);
		m_curr = ImageGPU<uchar4>(dim);

		auto output_dim = cuda::math::Div(dim, m_macroBlockDim);
		m_vecAndConf = ImageGPU<int4>(image::vec2ui{ output_dim.x, output_dim.y });
	}
}
