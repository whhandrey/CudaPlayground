#pragma once
#include <Cuda/Context.h>
#include <Image/ImageView.h>
#include "../BlockMatching.h"

namespace cuda {
	namespace motion {
		namespace visualization {
			using image::GpuImageView;

			void Conf(
				const GpuImageView<BlockMatchStats>& stats,
				GpuImageView<uchar4>& output,
				cuda::KernelContext& ctx,
				image::vec2ui blockDim = { 8, 8 }
			);

			void MotionMap(
				const GpuImageView<BlockMatchStats>& stats,
				GpuImageView<uchar4>& output,
				cuda::KernelContext& ctx,
				image::vec2i search_halfsize,
				image::vec2ui blockDim = { 8, 8 }
			);
		}
	}
}
