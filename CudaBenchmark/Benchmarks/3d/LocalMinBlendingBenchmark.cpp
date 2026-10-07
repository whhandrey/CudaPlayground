#include <Cuda/GpuImage/GpuImage.h>
#include <Cuda/GpuImage/GpuImageTransfer.h>
#include <Cuda/Memory/GpuBuffer.h>
#include <Cuda/Memory/GpuBufferTransfer.h>

#include <Image/Image.h>
#include <Image/ImageView.h>

#include <Cuda/KernelContext.h>
#include <Cuda/Profiler/GpuProfiler.h>

#include <Cuda/3d/LocalMinBlending.h>

#include "LocalMinBlendingBenchmark.h"
#include "../BenchOutput.h"
#include "../../Common/CudaStream.h"
#include "../../Common/RandomData.h"
#include "../../Common/NullProfiler.h"

#include <iomanip>

namespace {
	size_t CalcSharedMemFusedHalo(image::vec3ui blockDim, image::vec3i filter_halfsize) {
		const size_t tileSizeX = blockDim.x * blockDim.y * blockDim.z;
		const size_t tileSizeXY = blockDim.x * (blockDim.y - filter_halfsize.y * 2) * blockDim.z;

		const size_t sharedMemSize = (tileSizeX + tileSizeXY) * sizeof(float);
		return sharedMemSize;
	}

	std::string MakeSessionLabel(image::vec3ui blockDim, size_t sharedMemSize) {
		const auto formattedVec = bench::output::FormatVec3d(blockDim);
		if (sharedMemSize > 0) {
			return formattedVec + " " + std::to_string(sharedMemSize);
		}

		return formattedVec;
	}

	void WarmUpBlending3d(
		image::GpuVolumeView<float> baseData,
		image::GpuVolumeView<const float> gateField,
		image::GpuVolumeView<const float> activityField,
		image::GpuVolumeView<const float> filteredData,
		bool gateAlreadyProcessed,
		const cuda::erode3d::BlendParams& params,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::erode3d::BlendOp(
				baseData,
				gateField,
				activityField,
				filteredData,
				gateAlreadyProcessed,
				params,
				ctxNoProfile,
				{ 8, 8, 4 }
			);
		}
	}

	void BenchBlending3dImpl(
		image::GpuVolumeView<float> baseData,
		image::GpuVolumeView<const float> gateField,
		image::GpuVolumeView<const float> activityField,
		image::GpuVolumeView<const float> filteredData,
		bool gateAlreadyProcessed,
		const cuda::erode3d::BlendParams& params,
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

		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, 0));
				auto kernelCtx = cuda::KernelContext{ stream, &session };

				for (int i = 0; i < numRuns; ++i) {
					cuda::erode3d::BlendOp(
						baseData,
						gateField,
						activityField,
						filteredData,
						gateAlreadyProcessed,
						params,
						kernelCtx,
						blockDim
					);
				}
			}
		}
	}
}

namespace bench::localmin_blending3d {
	/*void CheckIfSepMultMatchesNaive() {
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
	}*/

	void Blending3dBenchmark() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		const image::vec3ui inputDim = { 601, 310, 169 };
		std::cout << std::endl << "Blending3dKernel of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> baseData = bench::data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> gateField = bench::data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> activityField = bench::data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> filteredData = bench::data::GenerateRandomVolume(inputDim);

		auto gpuBaseData = cuda::gpu_image::Create(baseData, stream.Get());
		const auto gpuGateField = cuda::gpu_image::Create(gateField, stream.Get());
		const auto gpuActivityField = cuda::gpu_image::Create(activityField, stream.Get());
		const auto gpuFilteredData = cuda::gpu_image::Create(filteredData, stream.Get());

		const auto params = cuda::erode3d::BlendParams {
			0.4f,  // gateThreshold
			5.0f,  // gateSlope: keep >= 0
			0.8f,  // alternateBlendWeight: keep > 0
			1.2f,  // defaultBlendWeight: keep > 0
			0.9f   // auxiliaryScale
		};

		bool gateAlreadyProcessed = false;

		WarmUpBlending3d(
			cuda::gpu_image::MakeVolumeView(gpuBaseData),
			cuda::gpu_image::MakeVolumeView(gpuGateField),
			cuda::gpu_image::MakeVolumeView(gpuActivityField),
			cuda::gpu_image::MakeVolumeView(gpuFilteredData),
			gateAlreadyProcessed,
			params,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();
		const int numIterations = 2;

		for (int i = 0; i < numIterations; ++i) {
			BenchBlending3dImpl(
				cuda::gpu_image::MakeVolumeView(gpuBaseData),
				cuda::gpu_image::MakeVolumeView(gpuGateField),
				cuda::gpu_image::MakeVolumeView(gpuActivityField),
				cuda::gpu_image::MakeVolumeView(gpuFilteredData),
				gateAlreadyProcessed,
				params,
				stream.Get(),
				profiler,
				runs
			);

			gateAlreadyProcessed = true;
		}

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}
}
