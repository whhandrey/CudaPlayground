#include "../Image/Image.h"

namespace motion {
	// simple version, no warp sync
	ImageGPU<int4> BlockMatchingSimple(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 macroBlockSize, int2 search_halfsize, ivec2 cudaBlockDim, cudaStream_t stream);

	ImageGPU<int4> BlockMatchingSimpleT(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 macroBlockSize, int2 search_halfsize, ivec2 cudaBlockDim, cudaStream_t stream);

	// not improved now since this is very slow
	ImageGPU<int2> BlockMatchingWarp(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 macroBlockSize, int2 search_halfsize, ivec2 cudaBlockDim, cudaStream_t stream);

	namespace shift {
		ImageGPU<uchar4> ShiftImage(const ImageGPU<uchar4>& image, int2 shiftVector, cudaStream_t stream, ivec2 blockSize = { 16, 16 });
	}
}
