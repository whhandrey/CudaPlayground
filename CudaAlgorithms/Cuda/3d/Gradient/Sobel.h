#pragma once

#include <Image/ImageView.h>
#include <Cuda/KernelContext.h>

namespace cuda {
	namespace processing3d {
		void SobelMagnitude(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			cuda::KernelContext& ctx
		);
	}
}
