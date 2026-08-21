#pragma once
#include <Cuda/KernelContext.h>
#include <Image/ImageView.h>

namespace cuda {
	namespace transform {
		using image::GpuImageView;

		void ShiftImage(
			const GpuImageView<uchar4>& image,
			GpuImageView<uchar4>& output,
			image::vec2i shiftVector,
			cuda::KernelContext& ctx,
			image::vec2ui blockSize = { 16, 16 }
		);
	}
}
