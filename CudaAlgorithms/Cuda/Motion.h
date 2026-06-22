#include <Cuda/Context.h>
#include <Image/ImageView.h>

namespace cuda {
	namespace motion {
		using image::GpuImageView;

		struct BlockMatchingParams {
			image::vec2ui blockDim;
			image::vec2ui macroBlockDim;
			image::vec2i search_halfsize;
		};

		image::vec2ui MotionOutputDim(image::vec2ui dim, image::vec2ui macroBlockDim);

		// simple version, no warp sync
		void BlockMatchingSimple(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<int4>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		// Only supports blockDim = 8;8, search_halfsize = 3;3, macroBlockDim = 16;16
		void BlockMatchingSimpleT(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<int4>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		// not improved now since this is very slow comp. to simple version

		void BlockMatchingWarp(
			const GpuImageView<uchar4>& prevFrame,
			const GpuImageView<uchar4>& currFrame,
			GpuImageView<int4>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		namespace shift {
			void ShiftImage(
				const GpuImageView<uchar4>& image,
				GpuImageView<uchar4>& output,
				image::vec2i shiftVector,
				cuda::KernelContext& ctx,
				image::vec2ui blockSize = { 16, 16 }
			);
		}
	}
}
