#include <Cuda/Context.h>
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
			const image::GpuImageView<image::vec4uc>& input,
			image::GpuImageView<image::vec4f>& tmp_buffer,
			image::GpuImageView<image::vec4uc>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		// convention: blockDim = { x, y + 2 * filter_halfsize }, useful output = { x, y }
		void GaussianBlurFusedV1(
			const image::GpuImageView<image::vec4uc>& input,
			image::GpuImageView<image::vec4uc>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		// convention: blockDim = { x, y }, useful output = { x, y - 2 * filter_halfsize }
		void GaussianBlurFusedV2(
			const image::GpuImageView<image::vec4uc>& input,
			image::GpuImageView<image::vec4uc>& output,
			const GaussianBlurParams& p,
			cuda::KernelContext& ctx
		);

		void BilateralFilter(
			const image::GpuImageView<image::vec4uc>& input,
			image::GpuImageView<image::vec4uc>& output,
			const BilateralParams& p,
			cuda::KernelContext& ctx
		);

		void BilateralFilterT(
			const image::GpuImageView<image::vec4uc>& input,
			image::GpuImageView<image::vec4uc>& output,
			const BilateralParams& p,
			cuda::KernelContext& ctx,
			bool debugInfo = false
		);
	}
}
