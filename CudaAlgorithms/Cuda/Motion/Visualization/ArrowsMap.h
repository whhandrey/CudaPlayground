#pragma once
#include <Cuda/KernelContext.h>
#include <Image/ImageView.h>
#include "../BlockMatching.h"

namespace cuda {
	namespace motion {
		namespace visualization {
			using image::GpuImageView;

			// this uses anti-aliasing method of making a capsule around the arrow
			// and using coverage to blend colors at the edges of arrows

			// TODO: this kernel is a bit slow, it needs probably:
			// 1. Reduction for aggregate pass
			// 2. One cuda block handles one arrow, not like now (one thread per arrow)
			void ArrowsMap(
				const GpuImageView<BlockMatchStats>& stats,
				GpuImageView<uchar4>& output,
				cuda::KernelContext& ctx,
				image::vec2ui macroBlockDim,
				int groupSize,
				float thickness = 1.25f,
				image::vec2ui blockDim = { 16, 16 }
			);
		}
	}
}
