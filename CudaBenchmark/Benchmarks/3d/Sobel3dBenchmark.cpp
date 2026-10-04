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
	size_t CalcSharedMemSize(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
		const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
		const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

		return tileWidth * tileHeight * tileDepth * sizeof(float);
	}

	std::string MakeSessionLabel(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		return bench::output::FormatVec3d(blockDim) + " " + std::to_string(CalcSharedMemSize(blockDim, filter_halfsize));
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
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, filter_halfsize));
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
}

namespace bench {
	//void CheckIfBothMatch() {
	//	CudaStream stream;

	//	auto dummyProfiler = bench::NullProfiler();
	//	auto ctxNoProfile = cuda::KernelContext{ stream.Get(), &dummyProfiler };

	//	image::vec3ui inputDim = { 601, 310, 168 };
	//	image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

	//	image::vec3i filter_halfsize = { 7, 5, 1 };

	//	std::vector<float> weightsX = bench::data::GenerateRandomWeights(filter_halfsize.x);
	//	std::vector<float> weightsY = bench::data::GenerateRandomWeights(filter_halfsize.y);
	//	std::vector<float> weightsZ = bench::data::GenerateRandomWeights(filter_halfsize.z);

	//	const int nx = 2 * filter_halfsize.x + 1;
	//	const int ny = 2 * filter_halfsize.y + 1;
	//	const int nz = 2 * filter_halfsize.z + 1;

	//	std::vector<float> weights(nx * ny * nz);

	//	for (int z = 0; z < nz; ++z) {
	//		for (int y = 0; y < ny; ++y) {
	//			for (int x = 0; x < nx; ++x) {
	//				weights[x + y * nx + z * nx * ny] = weightsX[x] * weightsY[y] * weightsZ[z];
	//			}
	//		}
	//	}

	//	const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
	//	const auto weights_buf = cuda::gpu_buffer::Upload(weights, stream.Get());
	//	cuda::gpu_image::GpuVolume<float> output_naive(input_gpu.Dim());

	//	cuda::conv3d::Conv3dNaiveSharedMem(
	//		cuda::gpu_image::MakeVolumeView(input_gpu),
	//		cuda::gpu_buffer::MakeSpan(weights_buf),
	//		cuda::gpu_image::MakeVolumeView(output_naive),
	//		filter_halfsize,
	//		ctxNoProfile,
	//		{ 32, 4, 2 }
	//	);

	//	cuda::gpu_image::GpuVolume<float> output_fused(input_gpu.Dim());
	//	const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
	//	const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
	//	const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

	//	cuda::conv3d::Conv3dFusedSeparable(
	//		cuda::gpu_image::MakeVolumeView(input_gpu),
	//		cuda::gpu_buffer::MakeSpan(weightsX_buf),
	//		cuda::gpu_buffer::MakeSpan(weightsY_buf),
	//		cuda::gpu_buffer::MakeSpan(weightsZ_buf),
	//		cuda::gpu_image::MakeVolumeView(output_fused),
	//		filter_halfsize,
	//		ctxNoProfile,
	//		{ 16, 32, 1 }
	//	);

	//	const auto naive_cpu = cuda::gpu_image::Download(output_naive, stream.Get());
	//	const auto fused_cpu = cuda::gpu_image::Download(output_fused, stream.Get());

	//	cudaCheck(cudaStreamSynchronize(stream.Get()));

	//	size_t failedCount = 0;
	//	size_t maxDiffIndex = 0;
	//	double maxAbsDiff = 0.0;

	//	std::vector<size_t> failedIndices;

	//	for (size_t i = 0; i < naive_cpu.m_data.size(); ++i) {
	//		const double a = naive_cpu.m_data[i];
	//		const double b = fused_cpu.m_data[i];

	//		const double diff = std::abs(a - b);
	//		const double tolerance = 2e-5 + 1e-4 * std::max(std::abs(a), std::abs(b));

	//		if (diff > tolerance) {
	//			++failedCount;
	//			failedIndices.push_back(i);
	//		}

	//		if (diff > maxAbsDiff) {
	//			maxAbsDiff = diff;
	//			maxDiffIndex = i;
	//		}
	//	}

	//	std::cout << std::setprecision(10)
	//		<< "Failed: " << failedCount
	//		<< " / " << naive_cpu.m_data.size() << '\n'
	//		<< "Max absolute difference (finite pairs): " << maxAbsDiff << '\n';

	//	if (maxAbsDiff > 0.0) {
	//		const size_t x = maxDiffIndex % inputDim.x;
	//		const size_t y = (maxDiffIndex / inputDim.x) % inputDim.y;
	//		const size_t z =
	//			maxDiffIndex / (size_t(inputDim.x) * inputDim.y);

	//		std::cout << "At index " << maxDiffIndex
	//			<< " -> (" << x << ", " << y << ", " << z << ")\n"
	//			<< "Naive: " << naive_cpu.m_data[maxDiffIndex] << '\n'
	//			<< "Fused: " << fused_cpu.m_data[maxDiffIndex] << '\n';
	//	}

	//	std::cout << std::endl;

	//	for (const size_t idx : failedIndices) {
	//		const size_t x = idx % inputDim.x;
	//		const size_t y = (idx / inputDim.x) % inputDim.y;
	//		const size_t z = idx / (size_t(inputDim.x) * inputDim.y);

	//		std::cout << "Failed at index " << idx
	//			<< " -> (" << x << ", " << y << ", " << z << ")\n"
	//			<< "Naive: " << naive_cpu.m_data[idx] << '\n'
	//			<< "Fused: " << fused_cpu.m_data[idx] << '\n';
	//	}
	//}

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
}
