#include <Cuda/KernelContext.h>
#include <Image/ImageTypes.h>
#include <Image/ImageView.h>

namespace cuda {
	namespace filter {
		struct GaussianBlurParams {
			image::vec2ui blockDim;
			int filter_halfsize;
			float sigma;
		};

		struct BilateralParams {
			image::vec2ui blockDim;
			int filter_halfsize;
			float sigmaGauss;
			float sigmaColor;
		};

		void GaussianBlur(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<float4>& tmp_buffer,
			image::GpuImageView<uchar4>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		// convention: blockDim = { x, y + 2 * filter_halfsize }, useful output = { x, y }
		void GaussianBlurFusedV1(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		// convention: blockDim = { x, y }, useful output = { x, y - 2 * filter_halfsize }
		void GaussianBlurFusedV2(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		void BilateralFilter(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			const BilateralParams& p,
			cuda::KernelContext& ctx
		);

		void BilateralFilterT(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			const BilateralParams& p,
			cuda::KernelContext& ctx,
			bool debugInfo = false
		);
	}
}
