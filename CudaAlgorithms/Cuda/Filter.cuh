#include "../Image/Image.h"

namespace filter {
	ImageGPU<uchar4> GaussianBlur(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, ivec2 blockSize, cudaStream_t stream);

	// convention: blockDim = { x, y + 2 * filter_halfsize }, useful output = { x, y }
	ImageGPU<uchar4> GaussianBlurFusedV1(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, ivec2 blockSize, cudaStream_t stream);
	
	// convention: blockDim = { x, y }, useful output = { x, y - 2 * filter_halfsize }
	ImageGPU<uchar4> GaussianBlurFusedV2(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, ivec2 blockSize, cudaStream_t stream);

	ImageGPU<uchar4> BilateralFilter(const ImageGPU<uchar4>& input, int filter_halfsize, float sigmaGauss, float sigmaColor, ivec2 blockSize, cudaStream_t stream);

	ImageGPU<uchar4> BilateralFilterT(const ImageGPU<uchar4>& input, float sigmaGauss, float sigmaColor, ivec2 blockSize, cudaStream_t stream, bool debugInfo = false);
}
