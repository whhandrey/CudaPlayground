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

#include <iomanip>

namespace {
	void WarmUpNaive(
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

	void BenchNaiveImpl(
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

	void WarmUpFusedSeparable(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weightsX,
		memory::GpuSpan<const float> weightsY,
		memory::GpuSpan<const float> weightsZ,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		cuda::KernelContext ctx,
		int warmUpRuns)
	{
		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::conv3d::Conv3dFusedSeparable(input, weightsX, weightsY, weightsZ, output, filter_halfsize, ctx, { 8, 8, 8 });
		}
	}

	void BenchFusedSeparableImpl(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weightsX,
		memory::GpuSpan<const float> weightsY,
		memory::GpuSpan<const float> weightsZ,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		const CudaStream& stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		const std::vector<image::vec3ui> blockDims {
			// 64 threads
			{  8,  8, 1 },
			{ 16,  4, 1 },
			{ 32,  2, 1 },

			// 128 threads
			{  8, 16, 1 },
			{ 16,  8, 1 },
			{ 32,  4, 1 },
			{ 64,  2, 1 },

			// 256 threads
			{  8, 32, 1 },
			{ 16, 16, 1 },
			{ 32,  8, 1 },
			{ 64,  4, 1 },

			// 512 threads
			{ 16, 32, 1 },
			{ 32, 16, 1 },
			{ 64,  8, 1 }
		};

		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, filter_halfsize));
				auto kernelCtx = cuda::KernelContext{ stream.Get(), &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::conv3d::Conv3dFusedSeparable(input, weightsX, weightsY, weightsZ, output, filter_halfsize, kernelCtx, blockDim);
				}
			}
		}
	}

	void WarmUpFusedSeparableMultipleOutputs(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weightsX,
		memory::GpuSpan<const float> weightsY,
		memory::GpuSpan<const float> weightsZ,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		int numOutputs,
		cuda::KernelContext ctx,
		int warmUpRuns)
	{
		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::conv3d::Conv3dFusedSeparableMultipleOutputs(
				input,
				weightsX,
				weightsY,
				weightsZ,
				output,
				filter_halfsize,
				numOutputs,
				ctx,
				{ 8, 8, 8 });
		}
	}

	void BenchFusedSeparableMultipleOutputsImpl(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> weightsX,
		memory::GpuSpan<const float> weightsY,
		memory::GpuSpan<const float> weightsZ,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		int numOutputs,
		const CudaStream& stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		const std::vector<image::vec3ui> blockDims {
			// 64 threads
			{  8,  8, 1 },
			{ 16,  4, 1 },
			{ 32,  2, 1 },

			// 128 threads
			{  8, 16, 1 },
			{ 16,  8, 1 },
			{ 32,  4, 1 },
			{ 64,  2, 1 },

			// 256 threads
			{  8, 32, 1 },
			{ 16, 16, 1 },
			{ 32,  8, 1 },
			{ 64,  4, 1 },

			// 512 threads
			{ 16, 32, 1 },
			{ 32, 16, 1 },
			{ 64,  8, 1 }
		};

		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, filter_halfsize));
				auto kernelCtx = cuda::KernelContext{ stream.Get(), &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::conv3d::Conv3dFusedSeparableMultipleOutputs(
						input,
						weightsX,
						weightsY,
						weightsZ,
						output,
						filter_halfsize,
						numOutputs,
						kernelCtx,
						blockDim);
				}
			}
		}
	}
}

namespace bench {
	void CheckIfBothMatch() {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 168 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 7, 5, 1 };

		std::vector<float> weightsX = bench::data::GenerateRandomWeights(filter_halfsize.x);
		std::vector<float> weightsY = bench::data::GenerateRandomWeights(filter_halfsize.y);
		std::vector<float> weightsZ = bench::data::GenerateRandomWeights(filter_halfsize.z);

		const int nx = 2 * filter_halfsize.x + 1;
		const int ny = 2 * filter_halfsize.y + 1;
		const int nz = 2 * filter_halfsize.z + 1;

		std::vector<float> weights(nx * ny * nz);

		for (int z = 0; z < nz; ++z) {
			for (int y = 0; y < ny; ++y) {
				for (int x = 0; x < nx; ++x) {
					weights[x + y * nx + z * nx * ny] = weightsX[x] * weightsY[y] * weightsZ[z];
				}
			}
		}

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		const auto weights_buf = cuda::gpu_buffer::Upload(weights, stream.Get());
		cuda::gpu_image::GpuVolume<float> output_naive(input_gpu.Dim());

		cuda::conv3d::Conv3dNaiveSharedMem(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weights_buf),
			cuda::gpu_image::MakeVolumeView(output_naive),
			filter_halfsize,
			ctxNoProfile,
			{ 32, 4, 2 }
		);

		cuda::gpu_image::GpuVolume<float> output_fused(input_gpu.Dim());
		const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
		const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
		const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

		cuda::conv3d::Conv3dFusedSeparable(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weightsX_buf),
			cuda::gpu_buffer::MakeSpan(weightsY_buf),
			cuda::gpu_buffer::MakeSpan(weightsZ_buf),
			cuda::gpu_image::MakeVolumeView(output_fused),
			filter_halfsize,
			ctxNoProfile,
			{ 16, 32, 1 }
		);

		const auto naive_cpu = cuda::gpu_image::Download(output_naive, stream.Get());
		const auto fused_cpu = cuda::gpu_image::Download(output_fused, stream.Get());

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		size_t failedCount = 0;
		size_t maxDiffIndex = 0;
		double maxAbsDiff = 0.0;

		std::vector<size_t> failedIndices;

		for (size_t i = 0; i < naive_cpu.m_data.size(); ++i) {
			const double a = naive_cpu.m_data[i];
			const double b = fused_cpu.m_data[i];

			const double diff = std::abs(a - b);
			const double tolerance = 2e-5 + 1e-4 * std::max(std::abs(a), std::abs(b));

			if (diff > tolerance) {
				++failedCount;
				failedIndices.push_back(i);
			}

			if (diff > maxAbsDiff) {
				maxAbsDiff = diff;
				maxDiffIndex = i;
			}
		}

		std::cout << std::setprecision(10)
			<< "Failed: " << failedCount
			<< " / " << naive_cpu.m_data.size() << '\n'
			<< "Max absolute difference (finite pairs): " << maxAbsDiff << '\n';

		if (maxAbsDiff > 0.0) {
			const size_t x = maxDiffIndex % inputDim.x;
			const size_t y = (maxDiffIndex / inputDim.x) % inputDim.y;
			const size_t z =
				maxDiffIndex / (size_t(inputDim.x) * inputDim.y);

			std::cout << "At index " << maxDiffIndex
				<< " -> (" << x << ", " << y << ", " << z << ")\n"
				<< "Naive: " << naive_cpu.m_data[maxDiffIndex] << '\n'
				<< "Fused: " << fused_cpu.m_data[maxDiffIndex] << '\n';
		}

		std::cout << std::endl;

		for (const size_t idx : failedIndices) {
			const size_t x = idx % inputDim.x;
			const size_t y = (idx / inputDim.x) % inputDim.y;
			const size_t z = idx / (size_t(inputDim.x) * inputDim.y);

			std::cout << "Failed at index " << idx
				<< " -> (" << x << ", " << y << ", " << z << ")\n"
				<< "Naive: " << naive_cpu.m_data[idx] << '\n'
				<< "Fused: " << fused_cpu.m_data[idx] << '\n';
		}
	}

	void Conv3dNaiveSharedMemBench() {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 168 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 7, 5, 1 };
		std::vector<float> weights = bench::data::GenerateRandomWeights3d(filter_halfsize);

		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		std::cout << std::endl << "Conv3dNaiveSharedMem of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		const auto weights_buf = cuda::gpu_buffer::Upload(weights, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		const auto weights_span = cuda::gpu_buffer::MakeSpan(weights_buf);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		WarmUpNaive(input_view, weights_span, output_view, filter_halfsize, ctxNoProfile, warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchNaiveImpl(input_view, weights_span, output_view, filter_halfsize, stream, profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void Conv3dFusedSeparableBench() {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 168 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 7, 5, 1 };

		std::vector<float> weightsX = bench::data::GenerateRandomWeights(filter_halfsize.x);
		std::vector<float> weightsY = bench::data::GenerateRandomWeights(filter_halfsize.y);
		std::vector<float> weightsZ = bench::data::GenerateRandomWeights(filter_halfsize.z);

		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		std::cout << std::endl << "Conv3dFusedSeparable of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
		const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
		const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		const auto weightsX_span = cuda::gpu_buffer::MakeSpan(weightsX_buf);
		const auto weightsY_span = cuda::gpu_buffer::MakeSpan(weightsY_buf);
		const auto weightsZ_span = cuda::gpu_buffer::MakeSpan(weightsZ_buf);

		WarmUpFusedSeparable(input_view, weightsX_span, weightsY_span, weightsZ_span, output_view, filter_halfsize, ctxNoProfile, warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchFusedSeparableImpl(input_view, weightsX_span, weightsY_span, weightsZ_span, output_view, filter_halfsize, stream, profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void Conv3dFusedSeparableMultipleOutputsBench(int numOutputs) {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 169 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 7, 5, 1 };

		std::vector<float> weightsX = bench::data::GenerateRandomWeights(filter_halfsize.x);
		std::vector<float> weightsY = bench::data::GenerateRandomWeights(filter_halfsize.y);
		std::vector<float> weightsZ = bench::data::GenerateRandomWeights(filter_halfsize.z);

		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		std::cout << std::endl << "Conv3dFusedSeparableMultipleOutputs of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;
		std::cout << "numOutputs: " << numOutputs << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
		const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
		const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		const auto weightsX_span = cuda::gpu_buffer::MakeSpan(weightsX_buf);
		const auto weightsY_span = cuda::gpu_buffer::MakeSpan(weightsY_buf);
		const auto weightsZ_span = cuda::gpu_buffer::MakeSpan(weightsZ_buf);

		WarmUpFusedSeparableMultipleOutputs(input_view, weightsX_span, weightsY_span, weightsZ_span, output_view, filter_halfsize, numOutputs, ctxNoProfile, warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchFusedSeparableMultipleOutputsImpl(input_view, weightsX_span, weightsY_span, weightsZ_span, output_view, filter_halfsize, numOutputs, stream, profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}
}
