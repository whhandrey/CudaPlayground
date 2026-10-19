#include <Cuda/GpuImage/GpuImage.h>
#include <Cuda/GpuImage/GpuImageTransfer.h>
#include <Cuda/Memory/GpuBuffer.h>
#include <Cuda/Memory/GpuBufferTransfer.h>

#include <Image/Image.h>
#include <Image/ImageView.h>

#include <Cuda/KernelContext.h>
#include <Cuda/Profiler/GpuProfiler.h>

#include <Cuda/3d/SyntheticVolumePipeline.h>

#include "SynthVolumePipelineBench.h"
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
	void WarmUpSynthVolumePipelineRef(
		image::GpuVolumeView<float> inputOutput,
		memory::GpuSpan<const float> coeffs3d,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineRef(
				inputOutput,
				coeffs3d,
				sobelX,
				sobelY,
				sobelZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				ctxNoProfile
			);
		}
	}

	void BenchSynthVolumePipelineRefImpl(
		image::GpuVolumeView<float> inputOutput,
		memory::GpuSpan<const float> coeffs3d,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		auto session = profiler.CreateSession("benchmark", std::string());
		auto kernelCtx = cuda::KernelContext{ stream, &session };

		for (int i = 0; i < numRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineRef(
				inputOutput,
				coeffs3d,
				sobelX,
				sobelY,
				sobelZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				kernelCtx
			);
		}
	}

	void WarmUpSynthVolumePipelineOptABit(
		image::GpuVolumeView<float> inputOutput,
		memory::GpuSpan<const float> coeffsX,
		memory::GpuSpan<const float> coeffsY,
		memory::GpuSpan<const float> coeffsZ,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineConv3dOptABit(
				inputOutput,
				coeffsX,
				coeffsY,
				coeffsZ,
				sobelX,
				sobelY,
				sobelZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				ctxNoProfile
			);
		}
	}

	void BenchSynthVolumePipelineOptABitImpl(
		image::GpuVolumeView<float> inputOutput,
		memory::GpuSpan<const float> coeffsX,
		memory::GpuSpan<const float> coeffsY,
		memory::GpuSpan<const float> coeffsZ,
		memory::GpuSpan<const float> sobelX,
		memory::GpuSpan<const float> sobelY,
		memory::GpuSpan<const float> sobelZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		auto session = profiler.CreateSession("benchmark", std::string());
		auto kernelCtx = cuda::KernelContext{ stream, &session };

		for (int i = 0; i < numRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineConv3dOptABit(
				inputOutput,
				coeffsX,
				coeffsY,
				coeffsZ,
				sobelX,
				sobelY,
				sobelZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				kernelCtx
			);
		}
	}

	void WarmUpSynthVolumePipelineOpt(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> coeffsX,
		memory::GpuSpan<const float> coeffsY,
		memory::GpuSpan<const float> coeffsZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		int warmUpRuns)
	{
		auto dummyProfiler = bench::NullProfiler();
		auto ctxNoProfile = cuda::KernelContext{ stream, &dummyProfiler };

		for (int i = 0; i < warmUpRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineOpt(
				input,
				coeffsX,
				coeffsY,
				coeffsZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				ctxNoProfile
			);
		}
	}

	void BenchSynthVolumePipelineOptImpl(
		image::GpuVolumeView<const float> input,
		memory::GpuSpan<const float> coeffsX,
		memory::GpuSpan<const float> coeffsY,
		memory::GpuSpan<const float> coeffsZ,
		cuda::erode3d::BlendParams params,
		cuda::synth_vol_pipeline::Workspace ws,
		image::vec3i conv3d_halfsize,
		image::vec3i erode3d_halfsize,
		cudaStream_t stream,
		cuda::profile::BasicGpuProfiler& profiler,
		int numRuns)
	{
		auto session = profiler.CreateSession("benchmark", std::string());
		auto kernelCtx = cuda::KernelContext{ stream, &session };

		for (int i = 0; i < numRuns; ++i) {
			cuda::synth_vol_pipeline::VolumePipelineOpt(
				input,
				coeffsX,
				coeffsY,
				coeffsZ,
				params,
				ws,
				conv3d_halfsize,
				erode3d_halfsize,
				kernelCtx
			);
		}
	}
}

namespace bench::synth_pipeline {
	void SynthVolPipelineBenchRef() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };

		const image::vec3i conv3d_halfsize = { 7, 5, 3 };
		const image::vec3i erode3d_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SynthVolumePipelineRef of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "conv3d_halfsize: " << bench::output::FormatVec3d(conv3d_halfsize) << std::endl;
		std::cout << "erode3d_halfsize: " << bench::output::FormatVec3d(erode3d_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		std::vector<float> weights = bench::data::GenerateRandomWeights3d(conv3d_halfsize);

		const auto [sobelX, sobelY, sobelZ] = SobelKernels3d();

		auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto sobelX_buf = cuda::gpu_buffer::Upload(sobelX, stream.Get());
		const auto sobelY_buf = cuda::gpu_buffer::Upload(sobelY, stream.Get());
		const auto sobelZ_buf = cuda::gpu_buffer::Upload(sobelZ, stream.Get());

		const auto weights_buf = cuda::gpu_buffer::Upload(weights, stream.Get());

		cuda::gpu_image::GpuVolume<float> tmp1(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> tmp2(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> filteredData(input_gpu.Dim());

		cuda::synth_vol_pipeline::Workspace ws = {
			cuda::gpu_image::MakeVolumeView(tmp1),
			cuda::gpu_image::MakeVolumeView(tmp2),
			cuda::gpu_image::MakeVolumeView(filteredData),
		};

		const auto params = cuda::erode3d::BlendParams {
			0.4f,  // gateThreshold
			5.0f,  // gateSlope: keep >= 0
			1.2f,  // blendWeight: keep > 0
			0.9f   // auxiliaryScale
		};

		WarmUpSynthVolumePipelineRef(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weights_buf),
			cuda::gpu_buffer::MakeSpan(sobelX_buf),
			cuda::gpu_buffer::MakeSpan(sobelY_buf),
			cuda::gpu_buffer::MakeSpan(sobelZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSynthVolumePipelineRefImpl(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weights_buf),
			cuda::gpu_buffer::MakeSpan(sobelX_buf),
			cuda::gpu_buffer::MakeSpan(sobelY_buf),
			cuda::gpu_buffer::MakeSpan(sobelZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			profiler,
			runs
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void SynthVolPipelineConv3dOptABitBench() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };

		const image::vec3i conv3d_halfsize = { 7, 5, 3 };
		const image::vec3i erode3d_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SynthVolumePipelineConv3dOptABit of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "conv3d_halfsize: " << bench::output::FormatVec3d(conv3d_halfsize) << std::endl;
		std::cout << "erode3d_halfsize: " << bench::output::FormatVec3d(erode3d_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		std::vector<float> weightsX = bench::data::GenerateRandomWeights(conv3d_halfsize.x);
		std::vector<float> weightsY = bench::data::GenerateRandomWeights(conv3d_halfsize.y);
		std::vector<float> weightsZ = bench::data::GenerateRandomWeights(conv3d_halfsize.z);

		const auto [sobelX, sobelY, sobelZ] = SobelKernels3d();

		auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto sobelX_buf = cuda::gpu_buffer::Upload(sobelX, stream.Get());
		const auto sobelY_buf = cuda::gpu_buffer::Upload(sobelY, stream.Get());
		const auto sobelZ_buf = cuda::gpu_buffer::Upload(sobelZ, stream.Get());

		const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
		const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
		const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

		cuda::gpu_image::GpuVolume<float> tmp1(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> tmp2(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> filteredData(input_gpu.Dim());

		cuda::synth_vol_pipeline::Workspace ws = {
			cuda::gpu_image::MakeVolumeView(tmp1),
			cuda::gpu_image::MakeVolumeView(tmp2),
			cuda::gpu_image::MakeVolumeView(filteredData),
		};

		const auto params = cuda::erode3d::BlendParams{
			0.4f,  // gateThreshold
			5.0f,  // gateSlope: keep >= 0
			1.2f,  // blendWeight: keep > 0
			0.9f   // auxiliaryScale
		};

		WarmUpSynthVolumePipelineOptABit(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weightsX_buf),
			cuda::gpu_buffer::MakeSpan(weightsY_buf),
			cuda::gpu_buffer::MakeSpan(weightsZ_buf),
			cuda::gpu_buffer::MakeSpan(sobelX_buf),
			cuda::gpu_buffer::MakeSpan(sobelY_buf),
			cuda::gpu_buffer::MakeSpan(sobelZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSynthVolumePipelineOptABitImpl(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weightsX_buf),
			cuda::gpu_buffer::MakeSpan(weightsY_buf),
			cuda::gpu_buffer::MakeSpan(weightsZ_buf),
			cuda::gpu_buffer::MakeSpan(sobelX_buf),
			cuda::gpu_buffer::MakeSpan(sobelY_buf),
			cuda::gpu_buffer::MakeSpan(sobelZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			profiler,
			runs
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}

	void SynthVolPipelineOptBench() {
		bench::output::PrintCudaDevice();
		bench::output::PrintSharedMemStats();

		image::vec3ui inputDim = { 601, 310, 169 };

		const image::vec3i conv3d_halfsize = { 7, 5, 3 };
		const image::vec3i erode3d_halfsize = { 1, 1, 1 };

		std::cout << std::endl << "SynthVolumePipelineOpt of volume of " << bench::output::FormatVec3d(inputDim) << std::endl;
		std::cout << "conv3d_halfsize: " << bench::output::FormatVec3d(conv3d_halfsize) << std::endl;
		std::cout << "erode3d_halfsize: " << bench::output::FormatVec3d(erode3d_halfsize) << std::endl;

		const int runs = 100;
		const int warmUpRuns = 20;

		CudaStream stream;
		image::CpuVolume<float> input = bench::data::GenerateRandomVolume(inputDim);

		std::vector<float> weightsX = bench::data::GenerateRandomWeights(conv3d_halfsize.x);
		std::vector<float> weightsY = bench::data::GenerateRandomWeights(conv3d_halfsize.y);
		std::vector<float> weightsZ = bench::data::GenerateRandomWeights(conv3d_halfsize.z);

		const auto input_gpu = cuda::gpu_image::Create(input, stream.Get());
		cuda::gpu_image::GpuVolume<float> output(input_gpu.Dim());

		const auto weightsX_buf = cuda::gpu_buffer::Upload(weightsX, stream.Get());
		const auto weightsY_buf = cuda::gpu_buffer::Upload(weightsY, stream.Get());
		const auto weightsZ_buf = cuda::gpu_buffer::Upload(weightsZ, stream.Get());

		cuda::gpu_image::GpuVolume<float> tmp1(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> tmp2(input_gpu.Dim());
		cuda::gpu_image::GpuVolume<float> filteredData(input_gpu.Dim());

		cuda::synth_vol_pipeline::Workspace ws = {
			cuda::gpu_image::MakeVolumeView(tmp1),
			cuda::gpu_image::MakeVolumeView(tmp2),
			cuda::gpu_image::MakeVolumeView(filteredData),
		};

		const auto params = cuda::erode3d::BlendParams{
			0.4f,  // gateThreshold
			5.0f,  // gateSlope: keep >= 0
			1.2f,  // blendWeight: keep > 0
			0.9f   // auxiliaryScale
		};

		WarmUpSynthVolumePipelineOpt(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weightsX_buf),
			cuda::gpu_buffer::MakeSpan(weightsY_buf),
			cuda::gpu_buffer::MakeSpan(weightsZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			warmUpRuns
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		auto profiler = cuda::profile::BasicGpuProfiler();

		BenchSynthVolumePipelineOptImpl(
			cuda::gpu_image::MakeVolumeView(input_gpu),
			cuda::gpu_buffer::MakeSpan(weightsX_buf),
			cuda::gpu_buffer::MakeSpan(weightsY_buf),
			cuda::gpu_buffer::MakeSpan(weightsZ_buf),
			params,
			ws,
			conv3d_halfsize,
			erode3d_halfsize,
			stream.Get(),
			profiler,
			runs
		);

		cudaCheck(cudaStreamSynchronize(stream.Get()));

		bench::output::PrintKernelStats(profiler.GetResults());
	}
}
