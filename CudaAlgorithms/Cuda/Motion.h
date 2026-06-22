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
			const GpuImageView<image::vec4uc>& prevFrame,
			const GpuImageView<image::vec4uc>& currFrame,
			GpuImageView<image::vec4i>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		// Only supports blockDim = 8;8, search_halfsize = 3;3, macroBlockDim = 16;16
		void BlockMatchingSimpleT(
			const GpuImageView<image::vec4uc>& prevFrame,
			const GpuImageView<image::vec4uc>& currFrame,
			GpuImageView<image::vec4i>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		// not improved now since this is very slow comp. to simple version

		void BlockMatchingWarp(
			const GpuImageView<image::vec4uc>& prevFrame,
			const GpuImageView<image::vec4uc>& currFrame,
			GpuImageView<image::vec4i>& output,
			const BlockMatchingParams& p,
			cuda::KernelContext& ctx
		);

		namespace shift {
			void ShiftImage(
				const GpuImageView<image::vec4uc>& image,
				GpuImageView<image::vec4uc>& output,
				image::vec2i shiftVector,
				cuda::KernelContext& ctx,
				image::vec2ui blockSize = { 16, 16 }
			);
		}
	}
}
