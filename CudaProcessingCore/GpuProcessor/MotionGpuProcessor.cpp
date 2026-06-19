#include "MotionGpuProcessor.h"
#include <Cuda/Motion.cuh>
#include <Cuda/MathUtils.h>

namespace {
	bool EqDim(image::vec2ui dim1, image::vec2ui dim2) {
		return dim1.x == dim2.x && dim1.y == dim2.y;
	}
}

namespace cuda_processing {
	Image<vec4uc> MotionGpuProcessor::Process(const ImageView<vec4uc>& prev, const ImageView<vec4uc>& curr) {
		if (!EqDim(prev.m_dim, curr.m_dim)) {
			throw std::logic_error("MotionGpuProcessor::Process: prev.dim != curr.dim");
		}

		AllocMem(prev.m_dim);
	}

	void MotionGpuProcessor::AllocMem(image::vec2ui dim) {
		const auto curr_dim = m_prev.Dim();

		if (curr_dim.x == dim.x && curr_dim.y == dim.y) {
			return;
		}

		m_prev = ImageGPU<vec4uc>(dim);
		m_curr = ImageGPU<vec4uc>(dim);

		auto output_dim = cuda::math::Div(dim, m_macroBlockDim);
		m_vecAndConf = ImageGPU<vec4i>(image::vec2ui{ output_dim.x, output_dim.y });
	}
}
