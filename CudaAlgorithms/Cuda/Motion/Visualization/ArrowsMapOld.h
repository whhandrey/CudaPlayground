#pragma once
#include <Cuda/Context.h>
#include <Image/ImageView.h>
#include "../BlockMatching.h"

namespace cuda {
	namespace motion {
		namespace visualization {
			using image::GpuImageView;

			// old method without anti-aliasing
			void ArrowsMapDDA(
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
