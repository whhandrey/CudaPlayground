#pragma once

#include <Image/ImageView.h>
#include <Memory/GpuSpan.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace conv3d {
		void Conv3dNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffs3d,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);

		void Conv3dFusedSeparable(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);

		void Conv3dFusedSeparableMultipleOutputs(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int numOutputs,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);
	}
}
