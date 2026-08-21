#pragma once
#include <Cuda/KernelContext.h>
#include <Image/ImageView.h>
#include "BlockMatchingParams.h"

namespace cuda {
	namespace motion {
		using image::GpuImageView;

		// not improved now since this is very slow comp. to simple version
		// but keeping it for reference/debugging/experiments later
		void BlockMatchingWarp(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<int4>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);
	}
}
	