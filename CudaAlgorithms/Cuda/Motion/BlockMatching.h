#pragma once

#include <Cuda/Context.h>
#include <Image/ImageView.h>
#include "BlockMatchingParams.h"

namespace cuda {
	namespace motion {
		using image::GpuImageView;

		// simple version, no warp sync
		void BlockMatching(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<unsigned char>& confOut,
			GpuImageView<int2>& dxdyOut,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		// Only supports blockDim = 8;8, search_halfsize = 3;3, macroBlockDim = 16;16
		// Params are not used yet, maybe will be extended templated version
		// TODO: make dispatcher
		void BlockMatchingSimpleT(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<unsigned char>& confOut,
			GpuImageView<int2>& dxdyOut,
			const BlockMatchingParams& /*p*/,
			cuda::KernelContext& ctx
		);
	}
}
