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

		struct BlendParams {
			float gateThreshold;
			float gateSlope;

			float alternateBlendWeight;
			float defaultBlendWeight;

			float auxiliaryScale;
		};

		void BlendOp(
			image::GpuVolumeView<float> baseData,
			image::GpuVolumeView<const float> gateField,
			image::GpuVolumeView<const float> activityField,
			image::GpuVolumeView<const float> filteredData,
			bool gateAlreadyProcessed,
			const BlendParams& params,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);

		void LocalMin3dThenBlendOp(
			image::GpuVolumeView<float> baseData,
			image::GpuVolumeView<const float> gateField,
			image::GpuVolumeView<const float> activityField,
			image::GpuVolumeView<float> filteredData,
			bool gateAlreadyProcessed,
			const BlendParams& params,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDimLocMin,
			image::vec3ui blockDimBlend
		);
	}
}
