#pragma once

#include <Image/ImageView.h>
#include <Memory/GpuSpan.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace erode3d {
		void LocalMin3dFusedHalo(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);

		struct MixingParams {
			bool compressedIntensity;
			float weightIntThresh;
			float weightIntSigm;
			float mixWeightCompr;
			float mixWeight;
			float compensation3d;
		};

		void MixingOpNaive(
			image::GpuVolumeView<float> inputOutput,
			image::GpuVolumeView<const float> log10Input,
			image::GpuVolumeView<const float> gradData,
			image::GpuVolumeView<const float> erodedData,
			const MixingParams& params,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);
	}
}
