#include "../DeviceImage/ImageGPU.h"
#include <Cuda/Context.h>

namespace motion {
	// simple version, no warp sync
	ImageGPU<image::vec4i> BlockMatchingSimple(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, image::vec2ui macroBlockSize, image::vec2i search_halfsize, image::vec2ui cudaBlockDim, cuda::KernelContext& ctx);

	ImageGPU<image::vec4i> BlockMatchingSimpleT(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, image::vec2ui macroBlockSize, image::vec2i search_halfsize, image::vec2ui cudaBlockDim, cuda::KernelContext& ctx);

	// not improved now since this is very slow
	ImageGPU<image::vec2i> BlockMatchingWarp(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, image::vec2ui macroBlockSize, image::vec2i search_halfsize, image::vec2ui cudaBlockDim, cuda::KernelContext& ctx);

	namespace shift {
		ImageGPU<uchar4> ShiftImage(const ImageGPU<uchar4>& image, image::vec2i shiftVector, cuda::KernelContext& ctx, image::vec2ui blockSize = { 16, 16 });
	}
}
