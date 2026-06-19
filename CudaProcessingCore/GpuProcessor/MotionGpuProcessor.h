#pragma once
#include "IMotionGpuProcessor.h"
#include "../DeviceImage/ImageGPU.h"

namespace cuda_processing {
	using image::vec4i;

	class MotionGpuProcessor : public IMotionGpuProcessor {
	public:
		Image<vec4uc> Process(const ImageView<vec4uc>& prev, const ImageView<vec4uc>& curr) override;

	private:
		void AllocMem(image::vec2ui dim);

	private:
		const image::vec2ui m_blockDim = { 8, 8 };
		const image::vec2ui m_macroBlockDim = { 16, 16 };
		const image::vec2i m_search_halfsize = { 3, 3 };

		ImageGPU<vec4uc> m_prev;
		ImageGPU<vec4uc> m_curr;
		ImageGPU<vec4i> m_vecAndConf;
	};
}
