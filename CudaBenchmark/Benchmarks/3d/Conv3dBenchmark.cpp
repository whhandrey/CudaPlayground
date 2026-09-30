#include <Cuda/GpuImage/GpuImage.h>
#include <Cuda/GpuImage/GpuImageTransfer.h>
#include <Cuda/Memory/GpuBuffer.h>
#include <Cuda/Memory/GpuBufferTransfer.h>

#include <Image/Image.h>
#include <Image/ImageView.h>

#include <Cuda/KernelContext.h>
#include <Cuda/Profiler/GpuProfiler.h>

#include <Cuda/3d/Conv3d.h>

#include "Conv3dBenchmark.h"
#include "../BenchOutput.h"
#include "../../Common/CudaStream.h"
#include "../../Common/RandomData.h"
#include "../../Common/NullProfiler.h"

namespace {
	void WarmUp(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weights,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		cuda::KernelContext ctx,
		int warmUpRuns)
	{
		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::conv3d::Conv3dNaiveSharedMem(input, weights, output, filter_halfsize, ctx, { 8, 8, 8 });
		}
	}

	size_t CalcSharedMemSize(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
		const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
		const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

		return tileWidth * tileHeight * tileDepth * sizeof(float);
	}

	std::string MakeSessionLabel(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		return bench::output::FormatVec3d(blockDim) + " " + std::to_string(CalcSharedMemSize(blockDim, filter_halfsize));
	}

	void BenchImpl(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weights,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		const CudaStream& stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		const std::vector<image::vec3ui> blockDims {
			// 128 threads
			{  8,  8, 2 },
			{ 16,  8, 1 },
			{ 32,  4, 1 },

			// 256 threads
			{  8,  8, 4 },
			{ 16,  8, 2 },
			{ 16, 16, 1 },
			{ 32,  4, 2 },
			{ 32,  8, 1 },
			{ 64,  4, 1 },

			// 512 threads
			{  8,  8, 8 },
			{ 16,  8, 4 },
			{ 16, 16, 2 },
			{ 32,  4, 4 },
			{ 32,  8, 2 },
			{ 32, 16, 1 },
			{ 64,  4, 2 }
		};

		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, filter_halfsize));
				auto kernelCtx = cuda::KernelContext{ stream.Get(), &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::conv3d::Conv3dNaiveSharedMem(input, weights, output, filter_halfsize, kernelCtx, blockDim);
				}
			}
		}
	}
}

namespace bench {
	void Conv3dBench() {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 168 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 7, 5, 1 };
		std::vector<float> weights = bench::data::GenerateRandomWeights(filter_halfsize);

		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		std::cout << std::endl << "Conv3d of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		const auto weights_buf = cuda::gpu_buffer::Upload(weights, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		const auto weights_span = cuda::gpu_buffer::MakeSpan(weights_buf);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		WarmUp(input_view, weights_span, output_view, filter_halfsize, ctxNoProfile, warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchImpl(input_view, weights_span, output_view, filter_halfsize, stream, profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}
}
