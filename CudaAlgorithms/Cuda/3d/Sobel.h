#pragma once

#include <Image/ImageView.h>
#include <Memory/GpuSpan.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace grad3d {
		void SobelMagNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			image::GpuVolumeView<float> output,
			cuda::KernelContext ctx,
			image::vec3ui blockDim
		);

		void SobelMagFusedSeparable(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			cuda::KernelContext ctx,
			image::vec3ui blockDim
		);
	}
}
