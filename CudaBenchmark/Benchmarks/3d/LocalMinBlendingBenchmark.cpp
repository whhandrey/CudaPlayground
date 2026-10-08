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

	std::string MakeCombinedSessionLabel(image::vec3ui blockDimLocMin, image::vec3ui blockDimBlend) {
		return bench::output::FormatVec3d(blockDimLocMin) + ":" + bench::output::FormatVec3d(blockDimBlend);
	}

	std::vector<image::vec3ui> LocalMinBlockDims(image::vec3i filter_halfsize) {
		if (filter_halfsize.x == 1 && filter_halfsize.y == 1 && filter_halfsize.z == 1) {
			return {
				// 256 threads
				{  8,  8, 4 },
				{ 16,  4, 4 },

				// 512 threads
				{  8,  8, 8 },
				{ 16,  8, 4 },
				{ 16,  4, 8 },
				{ 32,  4, 4 },

				// 768 threads
				{ 16,  8, 6 },
				{ 16,  6, 8 },

				// 1024 threads
				{ 16,  8, 8 },
				{  8, 16, 8 },
				{ 32,  8, 4 },
				{ 32,  4, 8 },

				{  8,  8, 10 },  // 640
				{  8, 10,  8 },  // 640
				{ 16,  5,  8 },  // 640
				{ 16,  8,  5 },  // 640
				{  8,  8, 12 }   // 768, also test for radius 1
			};
		}
		else if (filter_halfsize.x == 2 && filter_halfsize.y == 2 && filter_halfsize.z == 2) {
			return {
				// 512 threads
				{  8,  8, 8 },

				// 768 threads
				{ 16,  8, 6 },
				{  8, 16, 6 },
				{  8, 12, 8 },
				{  8,  8,12 },

				// 960–1024 threads
				{ 16, 10, 6 },
				{ 16,  8, 8 },
				{  8, 16, 8 },

				{  8,  8, 10 },
				{  8, 10,  8 },
				{  8, 10, 10 },
				{  8, 12, 10 },
			};
		}

		return {};
	}

	std::vector<image::vec3ui> Blending3dBlockDims() {
		return {
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
		const auto blockDims = Blending3dBlockDims();

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

	void WarmUpLocalMin3dFusedHalo(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::erode3d::LocalMin3dFusedHalo(input, output, filter_halfsize, ctxNoProfile, { 8, 8, 8 });
		}
	}

	void BenchLocalMin3dFusedHaloImpl(
		image::GpuVolumeView<const float> input,
		image::GpuVolumeView<float> output,
		image::vec3i filter_halfsize,
		cudaStream_t stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		const auto blockDims = LocalMinBlockDims(filter_halfsize);

		for (const auto& blockDim : blockDims)
		{
			// scoped session profile
			{
				const size_t sharedMemSize = CalcSharedMemFusedHalo(blockDim, filter_halfsize);
				auto session = profiler.CreateSession("benchmark", MakeSessionLabel(blockDim, sharedMemSize));

				auto kernelCtx = cuda::KernelContext{ stream, &session };
				for (int i = 0; i < numRuns; ++i) {
					cuda::erode3d::LocalMin3dFusedHalo(input, output, filter_halfsize, kernelCtx, blockDim);
				}
			}
		}
	}

	void WarmUpLocalMin3dAndBlending(
		image::GpuVolumeView<const float> baseData,
		image::GpuVolumeView<float> baseDataCpy,
		image::GpuVolumeView<const float> gateField,
		image::GpuVolumeView<const float> activityField,
		image::GpuVolumeView<float> filteredData,
		const cuda::erode3d::BlendParams& params,
		image::vec3i filter_halfsize,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			bool gateAlreadyProcessed = false;
			cuda::gpu_image::CopyVolumeAsync(baseData, baseDataCpy, stream);

			for (int i = 0; i < 2; ++i) {
				cuda::erode3d::LocalMin3dThenBlendOp(
					baseDataCpy,
					gateField,
					activityField,
					filteredData,
					gateAlreadyProcessed,
					params,
					filter_halfsize,
					ctxNoProfile,
					{ 8, 8, 8 },
					{ 8, 8, 4 }
				);

				gateAlreadyProcessed = true;
			}
		}
	}

	void BenchLocalMin3dThenBlend3dImpl(
		image::GpuVolumeView<const float> baseData,
		image::GpuVolumeView<float> baseDataCpy,
		image::GpuVolumeView<const float> gateField,
		image::GpuVolumeView<const float> activityField,
		image::GpuVolumeView<float> filteredData,
		const cuda::erode3d::BlendParams& params,
		image::vec3i filter_halfsize,
		cudaStream_t stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		// T1200
		//const std::vector<image::vec3ui> bestBlockDimsLocMinR1 {
		//	{ 8, 16, 8 },
		//	{ 16, 8, 8 }
		//};

		//const std::vector<image::vec3ui> bestBlockDimsLocMinR2 {
		//	{ 8, 12, 10 },
		//	{ 8, 16, 8 }
		//};

		// both T1200 and RTX 5060
		const std::vector<image::vec3ui> bestBlockDimsBlend {
			{ 32, 4, 1 },
			{ 64, 4, 1 }
		};

		// RTX 5060
		//const std::vector<image::vec3ui> bestBlockDimsLocMinR1 {
		//	{ 8, 8, 12 },
		//	{ 8, 10, 8 }
		//};

		const std::vector<image::vec3ui> bestBlockDimsLocMinR2 {
			{ 8, 8, 12 },
			{ 8, 12, 8 }
		};

		for (const auto& blockDimBlend : bestBlockDimsBlend)
		{
			for (const auto& blockDimLocMin : bestBlockDimsLocMinR2)
			{
				// scoped session profile
				{
					auto session = profiler.CreateSession("benchmark", MakeCombinedSessionLabel(blockDimLocMin, blockDimBlend));
					auto kernelCtx = cuda::KernelContext{ stream, &session };

					bool gateAlreadyProcessed = false;
					cuda::gpu_image::CopyVolumeAsync(baseData, baseDataCpy, stream);

					for (int i = 0; i < numRuns; ++i) {
						cuda::erode3d::LocalMin3dThenBlendOp(
							baseDataCpy,
							gateField,
							activityField,
							filteredData,
							gateAlreadyProcessed,
							params,
							filter_halfsize,
							kernelCtx,
							blockDimLocMin,
							blockDimBlend
						);
					}
				}
			}
		}
	}
}

namespace bench::localmin_blending3d {
	void Blending3dBenchmark() {
		output::PrintCudaDevice();
		output::PrintSharedMemStats();

		const image::vec3ui inputDim = { 601, 310, 169 };
		std::cout << std::endl << "Blending3dKernel of volume of " << output::FormatVec3d(inputDim) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> baseData = data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> gateField = data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> activityField = data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> filteredData = data::GenerateRandomVolume(inputDim);

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

		output::PrintKernelStats(profiler.GetResults());
	}

	void LocalMin3dBenchmark() {
		output::PrintCudaDevice();
		output::PrintSharedMemStats();

		const image::vec3ui inputDim = { 601, 310, 169 };
		const image::vec3i filter_halfsize{ 2, 2, 2 };

		std::cout << std::endl << "LocalMin3dFusedHalo of volume of " << output::FormatVec3d(inputDim) << std::endl;
		std::cout << std::endl << "filter_halfsize: " << output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = data::GenerateRandomVolume(inputDim);

		const auto gpuInput = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input.Dim());

		WarmUpLocalMin3dFusedHalo(
			cuda::gpu_image::MakeVolumeView(gpuInput),
			cuda::gpu_image::MakeVolumeView(output),
			filter_halfsize,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();
		BenchLocalMin3dFusedHaloImpl(
			cuda::gpu_image::MakeVolumeView(gpuInput),
			cuda::gpu_image::MakeVolumeView(output),
			filter_halfsize,
			stream.Get(),
			profiler,
			runs
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		output::PrintKernelStats(profiler.GetResults());
	}

	void LocalMin3dThenBlend3dBenchmark() {
		output::PrintCudaDevice();
		output::PrintSharedMemStats();

		const image::vec3ui inputDim = { 601, 310, 169 };
		const image::vec3i filter_halfsize = { 2, 2, 2 };

		std::cout << std::endl << "LocalMin3dThenBlend3d of volume of " << output::FormatVec3d(inputDim) << std::endl;
		std::cout << std::endl << "filter_halfsize: " << output::FormatVec3d(filter_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> baseData = data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> gateField = data::GenerateRandomVolume(inputDim);
		image::CpuVolume<float> activityField = data::GenerateRandomVolume(inputDim);

		const auto gpuBaseData = cuda::gpu_image::Create(baseData, stream.Get());
		cuda::gpu_image::GpuVolume<float> gpuBaseDataCpy(gpuBaseData.Dim());

		const auto gpuGateField = cuda::gpu_image::Create(gateField, stream.Get());
		const auto gpuActivityField = cuda::gpu_image::Create(activityField, stream.Get());
		cuda::gpu_image::GpuVolume<float> gpuFilteredData(gpuBaseData.Dim());

		const auto params = cuda::erode3d::BlendParams {
			0.4f,  // gateThreshold
			5.0f,  // gateSlope: keep >= 0
			0.8f,  // alternateBlendWeight: keep > 0
			1.2f,  // defaultBlendWeight: keep > 0
			0.9f   // auxiliaryScale
		};

		WarmUpLocalMin3dAndBlending(
			cuda::gpu_image::MakeVolumeView(gpuBaseData),
			cuda::gpu_image::MakeVolumeView(gpuBaseDataCpy),
			cuda::gpu_image::MakeVolumeView(gpuGateField),
			cuda::gpu_image::MakeVolumeView(gpuActivityField),
			cuda::gpu_image::MakeVolumeView(gpuFilteredData),
			params,
			filter_halfsize,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();
		BenchLocalMin3dThenBlend3dImpl(
			cuda::gpu_image::MakeVolumeView(gpuBaseData),
			cuda::gpu_image::MakeVolumeView(gpuBaseDataCpy),
			cuda::gpu_image::MakeVolumeView(gpuGateField),
			cuda::gpu_image::MakeVolumeView(gpuActivityField),
			cuda::gpu_image::MakeVolumeView(gpuFilteredData),
			params,
			filter_halfsize,
			stream.Get(),
			profiler,
			runs
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		output::PrintKernelStats(profiler.GetResults());
	}
}
