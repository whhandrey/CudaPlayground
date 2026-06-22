#pragma once
#include "IMotionGpuProcessor.h"
#include "../DeviceImage/ImageGPU.h"

namespace cuda_processing {
	class MotionGpuProcessor : public IMotionGpuProcessor {
	public:
		MotionGpuProcessor(cudaStream_t stream);

		Image<uchar4> Process(const ImageView<uchar4>& prev, const ImageView<uchar4>& curr) override;

	private:
		void AllocMem(image::vec2ui dim);

	private:
		cudaStream_t m_stream;

		const image::vec2ui m_blockDim = { 8, 8 };
		const image::vec2ui m_macroBlockDim = { 16, 16 };
		const image::vec2i m_search_halfsize = { 3, 3 };

		ImageGPU<uchar4> m_prev;
		ImageGPU<uchar4> m_curr;
		ImageGPU<int4> m_vecAndConf;
	};
}
