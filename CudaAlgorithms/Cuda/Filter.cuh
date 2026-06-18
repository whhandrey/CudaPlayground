#include "../DeviceImage/ImageGPU.h"
#include "../Context/Context.h"
#include <Image/ImageTypes.h>

namespace filter {
	using image::vec2ui;

	ImageGPU<uchar4> GaussianBlur(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx);

	// convention: blockDim = { x, y + 2 * filter_halfsize }, useful output = { x, y }
	ImageGPU<uchar4> GaussianBlurFusedV1(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx);
	
	// convention: blockDim = { x, y }, useful output = { x, y - 2 * filter_halfsize }
	ImageGPU<uchar4> GaussianBlurFusedV2(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx);

	ImageGPU<uchar4> BilateralFilter(const ImageGPU<uchar4>& input, int filter_halfsize, float sigmaGauss, float sigmaColor, vec2ui blockSize, cuda::KernelContext& ctx);

	ImageGPU<uchar4> BilateralFilterT(const ImageGPU<uchar4>& input, float sigmaGauss, float sigmaColor, vec2ui blockSize, cuda::KernelContext& ctx, bool debugInfo = false);
}
