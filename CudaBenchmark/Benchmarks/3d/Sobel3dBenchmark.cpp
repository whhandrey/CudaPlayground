#include <Cuda/GpuImage/GpuImage.h>
#include <Cuda/GpuImage/GpuImageTransfer.h>
#include <Cuda/Memory/GpuBuffer.h>
#include <Cuda/Memory/GpuBufferTransfer.h>

#include <Image/Image.h>
#include <Image/ImageView.h>

#include <Cuda/KernelContext.h>
#include <Cuda/Profiler/GpuProfiler.h>

#include <Cuda/3d/Sobel.h>

#include "Sobel3dBenchmark.h"
#include "../BenchOutput.h"
#include "../../Common/CudaStream.h"
#include "../../Common/RandomData.h"
#include "../../Common/NullProfiler.h"

#include <iomanip>

namespace {
	struct SobelKernels {
		std::vector<float> sobelX;
		std::vector<float> sobelY;
		std::vector<float> sobelZ;
	};

	SobelKernels SobelKernels3d() {
		constexpr float smooth[] = { 0.25f, 0.5f, 0.25f };
		constexpr float derivative[] = { -0.5f, 0.0f, 0.5f };
		const int sobelSize = 3;

		std::vector<float> sobelX;
		std::vector<float> sobelY;
		std::vector<float> sobelZ;

		for (int z = 0; z < sobelSize; ++z) {
			for (int y = 0; y < sobelSize; ++y) {
				for (int x = 0; x < sobelSize; ++x) {
					sobelX.push_back(derivative[x] * smooth[y] * smooth[z]);
					sobelY.push_back(smooth[x] * derivative[y] * smooth[z]);
					sobelZ.push_back(smooth[x] * smooth[y] * derivative[z]);
				}
			}
		}

		return { sobelX, sobelY, sobelZ };
	}
}

namespace {
	size_t CalcSharedMemNaive(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
		const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
		const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

		return tileWidth * tileHeight * tileDepth * sizeof(float);
	}

	size_t CalcSharedMemSeparale(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
		const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
		const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

		return tileWidth * tileHeight * tileDepth * sizeof(float);
	}

	std::string MakeSessionLabel(image::vec3ui blockDim, size_t sharedMemSize) {
		return bench::output::FormatVec3d(blockDim) + " " + std::to_string(sharedMemSize);
	}

	void WarmUpSobelNaive(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		image::GpuVolumeView<float> output,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::grad3d::SobelMagNaiveSharedMem(
				input,
				sobelX,
				sobelY,
				sobelZ,
				output,
				ctxNoProfile,
				{ 8, 8, 4 }
			);
		}
	}

	void BenchSobelNaiveImpl(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		image::GpuVolumeView<float> output,
		cudaStream_t stream,
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

		const image::vec3i filter_halfsize{ 1, 1, 1 };
		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, CalcSharedMemNaive(blockDim, filter_halfsize)));
				auto kernelCtx = cuda::KernelContext{ stream, &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::grad3d::SobelMagNaiveSharedMem(
						input,
						sobelX,
						sobelY,
						sobelZ,
						output,
						kernelCtx,
						blockDim
					);
				}
			}
		}
	}

	void WarmUpSobelFusedSeparable(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::grad3d::SobelMagFusedSeparable(input, output, ctxNoProfile, { 32, 4, 1 });
		}
	}

	void BenchSobelFusedSeparableImpl(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		cudaStream_t stream,
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

		const image::vec3i filter_halfsize{ 1, 1, 1 };
		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, CalcSharedMemSeparale(blockDim, filter_halfsize)));
				auto kernelCtx = cuda::KernelContext{ stream, &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::grad3d::SobelMagFusedSeparable(input, output, kernelCtx, blockDim);
				}
			}
		}
	}

	void WarmUpSobelFusedSepMultipleOutputs(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		int outputsPerThread,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::grad3d::SobelMagFusedSeparableMultipleOutputs(input, output, outputsPerThread, ctxNoProfile, { 8, 8, 8 });
		}
	}

	void BenchSobelFusedSepMultipleOutputsImpl(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		int outputsPerThread,
		cudaStream_t stream,
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

		const image::vec3i filter_halfsize{ 1, 1, 1 };
		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, CalcSharedMemSeparale(blockDim, filter_halfsize)));
				auto kernelCtx = cuda::KernelContext{ stream, &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::grad3d::SobelMagFusedSeparableMultipleOutputs(input, output, outputsPerThread, kernelCtx, blockDim);
				}
			}
		}
	}
}

namespace bench {
	void CheckIfSepMultMatchesNaive() {
		CudaStream stream;

		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

		image::vec3ui inputDim = { 601, 310, 169 };
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		image::vec3i filter_halfsize = { 1, 1, 1 };

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output_naive(input_gpu.Dim());

		const auto [sobelX, sobelY, sobelZ] = SobelKernels3d();

		const auto sobelX_buf = cuda::gpu_buffer::Upload(sobelX, stream.Get());
		const auto sobelY_buf = cuda::gpu_buffer::Upload(sobelY, stream.Get());
		const auto sobelZ_buf = cuda::gpu_buffer::Upload(sobelZ, stream.Get());

		cuda::grad3d::SobelMagNaiveSharedMem(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(sobelX_buf),
			cuda::gpu_buffer::MakeSpan(sobelY_buf),
			cuda::gpu_buffer::MakeSpan(sobelZ_buf),
			cuda::gpu_image::MakeVolumeView(output_naive),
			ctxNoProfile,
			{ 32, 8, 1 }
		);

		cuda::gpu_image::GpuVolume<float> output_fused(input_gpu.Dim());

		cuda::grad3d::SobelMagFusedSeparableMultipleOutputs(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_image::MakeVolumeView(output_fused),
			10,
			ctxNoProfile,
			{ 32, 8, 1 }
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

			const bool finiteA = std::isfinite(a);
			const bool finiteB = std::isfinite(b);

			if (!finiteA || !finiteB) {
				failedCount = 100500;
				break;
			}

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

	void SobelMag3dNaiveSharedMemBench() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };
		const image::vec3i filter_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SobelMagNaiveSharedMem of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		const auto [sobelX, sobelY, sobelZ] = SobelKernels3d();

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto sobelX_buf = cuda::gpu_buffer::Upload(sobelX, stream.Get());
		const auto sobelY_buf = cuda::gpu_buffer::Upload(sobelY, stream.Get());
		const auto sobelZ_buf = cuda::gpu_buffer::Upload(sobelZ, stream.Get());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		const auto sobelX_span = cuda::gpu_buffer::MakeSpan(sobelX_buf);
		const auto sobelY_span = cuda::gpu_buffer::MakeSpan(sobelY_buf);
		const auto sobelZ_span = cuda::gpu_buffer::MakeSpan(sobelZ_buf);

		WarmUpSobelNaive(input_view, sobelX_span, sobelY_span, sobelZ_span, output_view, stream.Get(), warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSobelNaiveImpl(input_view, sobelX_span, sobelY_span, sobelZ_span, output_view, stream.Get(), profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void SobelMag3dFusedSeparableBench() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };
		const image::vec3i filter_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SobelMagFusedSeparable of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		WarmUpSobelFusedSeparable(input_view, output_view, stream.Get(), warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSobelFusedSeparableImpl(input_view, output_view, stream.Get(), profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void SobelMag3dFusedSepMultipleOutputsBench(int outputsPerThread) {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };
		const image::vec3i filter_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SobelMag3dFusedSepMultipleOutputs of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "filter_halfsize: " << bench::output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto input_view = cuda::gpu_image::MakeVolumeView(input_gpu);
		auto output_view = cuda::gpu_image::MakeVolumeView(output);

		WarmUpSobelFusedSepMultipleOutputs(input_view, output_view, outputsPerThread, stream.Get(), warmUpRuns);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSobelFusedSepMultipleOutputsImpl(input_view, output_view, outputsPerThread, stream.Get(), profiler, runs);
		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}
}
