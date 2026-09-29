#pragma once

#include <Image/ImageView.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace grad3d {
		void SobelMagnitude(
			const image::GpuVolumeView<float>& input,
			image::GpuVolumeView<float>& output,
			cuda::KernelContext& ctx
		);
	}
}
