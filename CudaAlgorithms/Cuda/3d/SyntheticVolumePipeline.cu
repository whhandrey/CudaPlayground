#include "SyntheticVolumePipeline.h"
#include "Conv3d.h"
#include "Sobel.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <vector>

#include <Cuda/GpuImage/GpuImageView.h>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__ void Log10Kernel(
	image::GpuVolumeView<const float> input,
	image::GpuVolumeView<float> output)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= input.m_dim.x || y >= input.m_dim.y || z >= input.m_dim.z) {
		return;
	}

	const float sample = fmaxf(ReadPitched(input, x, y, z), 1e-6f);
	WritePitched(output, log10f(sample), x, y, z);
}

__global__ void SigmKernel(
	image::GpuVolumeView<float> inputOutput)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= inputOutput.m_dim.x || y >= inputOutput.m_dim.y || z >= inputOutput.m_dim.z) {
		return;
	}

	const float sampleOut = sigmoid(ReadPitched(inputOutput, x, y, z), 0.0f, 10.0f);
	WritePitched(inputOutput, sampleOut, x, y, z);
}

namespace detail {
	void VerifyInput(
		image::GpuVolumeView<const float> input,
		cuda::synth_vol_pipeline::Workspace ws)
	{
		if (input.m_dim != ws.tmp1.m_dim || input.m_dim != ws.tmp2.m_dim || input.m_dim != ws.filteredData.m_dim) {
			throw std::logic_error("CudaAlgoritms::SynthVolumePipeline: input/workspace dim mismatch");
		}
	}

	namespace ref {
		constexpr image::vec3ui conv3d_blockDim{ 8, 8, 4 };
		constexpr image::vec3ui conv3d_sep_blockDim{ 16, 32, 1 };

		constexpr image::vec3ui grad3d_blockDim{ 8, 8, 4 };
		constexpr image::vec3ui erode3d_blockDim{ 8, 8, 8 };
		constexpr image::vec3ui blend3d_blockDim{ 8, 8, 4 };
	}

	namespace opt {
		constexpr image::vec3ui conv3d_blockDim{ 16, 32, 1 };
		constexpr image::vec3ui grad3d_blockDim{ 32, 4, 1 };

		constexpr image::vec3ui erode3d_r1_blockDim{ 32, 8, 1 };
		constexpr image::vec3ui erode3d_r2_blockDim{ 16, 16, 2 };
	}
}

namespace cuda {
	namespace synth_vol_pipeline {
		void VolumePipelineRef(
			image::GpuVolumeView<float> inputOutput,
			memory::GpuSpan<const float> coeffs3d,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx)
		{
			detail::VerifyInput(cuda::gpu_image::ToConstView(inputOutput), ws);

			const int numIterations = 2;
			cuda::KernelContext enqueueCtx = ContextWithoutProfiling(ctx);

			const image::vec3ui blockDim = { 32, 8, 1 };
			const dim3 gridSize = cuda::math::DivUp(inputOutput.m_dim, blockDim);

			cuda::TimedCall("VolumePipelineRef", ctx, [&]() {
				for (int i = 0; i < numIterations; ++i) {
					cudaCheck(cudaPeekAtLastError());

					Log10Kernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (
						cuda::gpu_image::ToConstView(inputOutput),
						ws.tmp1
					);

					auto conv3dData = ws.tmp2;
					conv3d::Conv3dNaiveSharedMem(
						cuda::gpu_image::ToConstView(ws.tmp1),
						coeffs3d,
						conv3dData,
						conv3d_halfsize,
						enqueueCtx,
						detail::ref::conv3d_blockDim
					);

					// random iteration-dependent stuff
					if (i == 1) {
						SigmKernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (conv3dData);
					}

					auto gradData = ws.tmp1;
					grad3d::SobelMagNaiveSharedMem(
						cuda::gpu_image::ToConstView(conv3dData),
						sobelX,
						sobelY,
						sobelZ,
						gradData,
						enqueueCtx,
						detail::ref::grad3d_blockDim
					);

					erode3d::LocalMin3dFusedHalo(
						cuda::gpu_image::ToConstView(inputOutput),
						ws.filteredData,
						erode3d_halfsize,
						enqueueCtx,
						detail::ref::erode3d_blockDim
					);

					erode3d::BlendOp(
						inputOutput,
						cuda::gpu_image::ToConstView(conv3dData),
						cuda::gpu_image::ToConstView(gradData),
						cuda::gpu_image::ToConstView(ws.filteredData),
						blendParams,
						enqueueCtx,
						detail::ref::blend3d_blockDim
					);
				}
			});

			cudaCheck(cudaPeekAtLastError());
		}

		void VolumePipelineConv3dOptABit(
			image::GpuVolumeView<float> inputOutput,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx)
		{
			detail::VerifyInput(cuda::gpu_image::ToConstView(inputOutput), ws);

			const int numIterations = 2;
			cuda::KernelContext enqueueCtx = ContextWithoutProfiling(ctx);

			const image::vec3ui blockDim = { 32, 8, 1 };
			const dim3 gridSize = cuda::math::DivUp(inputOutput.m_dim, blockDim);

			cuda::TimedCall("VolumePipelineConv3dOptABit", ctx, [&]() {
				for (int i = 0; i < numIterations; ++i) {
					cudaCheck(cudaPeekAtLastError());

					Log10Kernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (
						cuda::gpu_image::ToConstView(inputOutput),
						ws.tmp1
					);

					auto conv3dData = ws.tmp2;
					conv3d::Conv3dFusedSeparable(
						cuda::gpu_image::ToConstView(ws.tmp1),
						coeffsX,
						coeffsY,
						coeffsZ,
						conv3dData,
						conv3d_halfsize,
						enqueueCtx,
						detail::ref::conv3d_sep_blockDim
					);

					// random iteration-dependent stuff
					if (i == 1) {
						SigmKernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (conv3dData);
					}

					auto gradData = ws.tmp1;
					grad3d::SobelMagNaiveSharedMem(
						cuda::gpu_image::ToConstView(conv3dData),
						sobelX,
						sobelY,
						sobelZ,
						gradData,
						enqueueCtx,
						detail::ref::grad3d_blockDim
					);

					erode3d::LocalMin3dFusedHalo(
						cuda::gpu_image::ToConstView(inputOutput),
						ws.filteredData,
						erode3d_halfsize,
						enqueueCtx,
						detail::ref::erode3d_blockDim
					);

					erode3d::BlendOp(
						inputOutput,
						cuda::gpu_image::ToConstView(conv3dData),
						cuda::gpu_image::ToConstView(gradData),
						cuda::gpu_image::ToConstView(ws.filteredData),
						blendParams,
						enqueueCtx,
						detail::ref::blend3d_blockDim
					);
				}
			});

			cudaCheck(cudaPeekAtLastError());
		}

		image::GpuVolumeView<float> VolumePipelineOpt(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx)
		{
			detail::VerifyInput(input, ws);

			const int numIterations = 2;
			cuda::KernelContext enqueueCtx = ContextWithoutProfiling(ctx);

			const image::vec3ui blockDim = { 32, 8, 1 };
			const dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const int outputsPerThreadConv3d = 13;
			const int outputsPerThreadSobel = 12;
			const int outputsPerThreadLocalMin3d = 9;

			cuda::TimedCall("VolumePipelineOpt", ctx, [&]() {
				Log10Kernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (input, ws.tmp1);
				cudaCheck(cudaPeekAtLastError());

				for (int i = 0; i < numIterations; ++i) {
					auto conv3dData = ws.tmp2;

					const bool runSigmoid = i == 1;

					// random iteration-dependent stuff
					const bool runLog10 = i + 1 < numIterations;

					conv3d::Conv3dFusedSeparableMultipleOutputs(
						cuda::gpu_image::ToConstView(ws.tmp1),
						coeffsX,
						coeffsY,
						coeffsZ,
						conv3dData,
						conv3d_halfsize,
						outputsPerThreadConv3d,
						runSigmoid,
						enqueueCtx,
						detail::opt::conv3d_blockDim
					);

					auto gradData = ws.tmp1;
					grad3d::SobelMagFusedSeparableMultipleOutputsV2(
						cuda::gpu_image::ToConstView(conv3dData),
						gradData,
						outputsPerThreadSobel,
						enqueueCtx,
						detail::opt::grad3d_blockDim
					);

					erode3d::LocalMin3dOptWithBlendOpFusedNew(
						input,
						cuda::gpu_image::ToConstView(conv3dData),
						cuda::gpu_image::ToConstView(gradData),
						ws.filteredData,
						blendParams,
						erode3d_halfsize,
						outputsPerThreadLocalMin3d,
						runLog10,
						enqueueCtx,
						detail::opt::erode3d_r1_blockDim
					);

					std::swap(ws.tmp1, ws.filteredData);
				}
			});

			cudaCheck(cudaPeekAtLastError());
			return ws.tmp1;
		}
	}
}
