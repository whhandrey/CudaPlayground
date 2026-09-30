#pragma once

#include <Image/ImageView.h>
#include <Memory/GpuSpan.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace conv3d {
		void Conv3dNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> weights,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim = { 8, 8, 8 }
		);
	}
}
