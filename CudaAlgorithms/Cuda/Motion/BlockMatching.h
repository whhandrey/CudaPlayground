#pragma once

#include <Cuda/Context.h>
#include <Image/ImageView.h>
#include "BlockMatchingParams.h"

namespace cuda {
	namespace motion {
		using image::GpuImageView;

		struct BlockMatchStats {
			image::vec2i bestDxDy;
			image::vec2i secondDxDy;

			int bestSad;
			int secondSad;

			int zeroSad;
			unsigned long sumSad;
		};

		// simple version, no warp sync
		void BlockMatching(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<BlockMatchStats>& statsOut,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);
	}
}
