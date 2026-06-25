#pragma once
#include <Cuda/Context.h>
#include <Cuda/Motion.h>
#include "IMotionGpuProcessor.h"
#include "../DeviceImage/ImageGPU.h"

namespace cuda {
	class MotionGpuProcessor : public IMotionGpuProcessor {
	public:
		MotionGpuProcessor();
		~MotionGpuProcessor();

		Image<unsigned char> Process(const ImageView<image::vec4uc>& prev, const ImageView<image::vec4uc>& curr) override;

	private:
		void AllocMem(image::vec2ui dim);

	private:
		cuda::KernelContext m_ctx;
		const cuda::motion::BlockMatchingParams m_params;

		ImageGPU<uchar4> m_prev;
		ImageGPU<uchar4> m_curr;
		ImageGPU<unsigned char> m_conf;
	};
}
