#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <Cuda/Image/ImageGPU.h>
#include <Cuda/Image/GpuImageTransfer.h>
#include <Image/ImageView.h>
#include <Cuda/Context.h>
#include <Cuda/Filter.h>
#include <Cuda/Motion/BlockMatching.h>
#include <Cuda/Transform/Shift.h>
#include <Profiler/Profiler.h>

#include <Image/Image.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <map>

#include "Profiler/Profiler.h"
#include "Image/ImageIO.h"

namespace stats {
	struct KernelRunStats {
		std::size_t count = 0;
		float avg = 0.0f;
		float median = 0.0f;
		float min = 0.0f;
		float max = 0.0f;
	};

	KernelRunStats CalcStats(const std::vector<float>& times) {
		KernelRunStats stats{};
		stats.count = times.size();

		if (times.empty()) {
			return stats;
		}

		auto [minIt, maxIt] = std::minmax_element(times.begin(), times.end());

		stats.min = *minIt;
		stats.max = *maxIt;

		double sum = std::accumulate(times.begin(), times.end(), 0.0);
		stats.avg = static_cast<float>(sum / static_cast<double>(times.size()));

		std::vector<float> sorted = times;
		std::sort(sorted.begin(), sorted.end());

		const std::size_t n = sorted.size();

		if (n % 2 == 1) {
			stats.median = sorted[n / 2];
		}
		else {
			stats.median = 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
		}

		return stats;
	}
}

namespace print {
	void PrintKernelStats(
		const std::map<std::string, std::vector<float>>& runs,
		std::ostream& os = std::cout)
	{
		os << std::fixed << std::setprecision(4);

		os << "\nCUDA kernel timings:\n\n";

		int kernelW = 0;
		for (const auto& [name, _] : runs) {
			kernelW = std::max(kernelW, static_cast<int>(name.size()));
		}

		kernelW = std::max(kernelW, 28) + 2;

		constexpr int runsW = 8;
		constexpr int avgW = 14;
		constexpr int medianW = 14;
		constexpr int minW = 14;
		constexpr int maxW = 14;

		const int totalW = kernelW + runsW + avgW + medianW + minW + maxW;

		os << std::left
			<< std::setw(kernelW) << "Kernel"
			<< std::right
			<< std::setw(runsW) << "Runs"
			<< std::setw(avgW) << "Avg ms"
			<< std::setw(medianW) << "Median ms"
			<< std::setw(minW) << "Min ms"
			<< std::setw(maxW) << "Max ms"
			<< '\n';

		os << std::string(totalW, '-') << '\n';

		for (const auto& [name, times] : runs) {
			stats::KernelRunStats stats = stats::CalcStats(times);

			os << std::left
				<< std::setw(kernelW) << name
				<< std::right
				<< std::setw(runsW) << stats.count
				<< std::setw(avgW) << stats.avg
				<< std::setw(medianW) << stats.median
				<< std::setw(minW) << stats.min
				<< std::setw(maxW) << stats.max
				<< '\n';
		}

		os << '\n';
	}
}

void PrintCudaDevice() {
	int device = 0;
	cudaGetDevice(&device); // current CUDA device

	cudaDeviceProp props{};
	cudaCheck(cudaGetDeviceProperties(&props, device));

	std::cout << "CUDA device: " << props.name << std::endl
		<< "sm_" << props.major << props.minor
		<< std::endl;
}

void PrintSharedMemStats() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::cout << "Shared memory per block: "
		<< prop.sharedMemPerBlock << " bytes\n";

	std::cout << "Shared memory per SM: "
		<< prop.sharedMemPerMultiprocessor << " bytes\n";

	std::cout << "SM count: "
		<< prop.multiProcessorCount << "\n";

	std::cout << "Max threads per SM: "
		<< prop.maxThreadsPerMultiProcessor << "\n";

	std::cout << "Warp size: "
		<< prop.warpSize << "\n";

	std::cout << "Max warps per SM: "
		<< prop.maxThreadsPerMultiProcessor / prop.warpSize
		<< "\n";

	std::cout << "Max registers per SM: "
		<< prop.regsPerMultiprocessor
		<< "\n";

	std::cout << "Max registers per block: "
		<< prop.regsPerBlock
		<< "\n";
}

int main() {
	cudaStream_t stream;
	cudaCheck(cudaStreamCreate(&stream));

	auto profiler = std::make_unique<cuda::profile::SampledProfiler>();

	cuda::KernelContext ctx {
		stream,
		profiler.get()
	};

	const std::string file = "C:\\AY\\Code\\ImgTest\\1.jpg";

	image::Image<uchar4> img = image::LoadFromFile<uchar4>(file);

	cuda::gpu_image::ImageGPU<uchar4> prevFrame = cuda::gpu_image::Create(img, stream);
	cuda::gpu_image::ImageGPU<uchar4> currFrame = cuda::gpu_image::Create(img, stream);
	cuda::gpu_image::ImageGPU<uchar4> currFrameShifted(img.Dim());
	cuda::gpu_image::ImageGPU<cuda::motion::BlockMatchStats> output(img.Dim());

	image::GpuImageView<uchar4> prevView{ prevFrame.Data(), prevFrame.Dim(), prevFrame.Pitch() };
	image::GpuImageView<uchar4> currView{ currFrame.Data(), currFrame.Dim(), currFrame.Pitch() };
	image::GpuImageView<uchar4> currShiftedView{ currFrameShifted.Data(), currFrameShifted.Dim(), currFrameShifted.Pitch() };
	image::GpuImageView<cuda::motion::BlockMatchStats> outView{ output.Data(), output.Dim(), output.Pitch() };

	cuda::transform::ShiftImage(currView, currShiftedView, { -3, 2 }, ctx);

	PrintCudaDevice();
	PrintSharedMemStats();

	std::cout << std::endl << "Testing image of (" << img.Dim().x << "; " << img.Dim().y << ")" << std::endl;

	image::vec2i search_halfsize = { 3, 3 };
	image::vec2ui macroBlockDim = { 16, 16 };

	std::cout
		<< "search_halfsize: (" << search_halfsize.x << "; " << search_halfsize.y << ")\t"
		<< "macroBlockDim: (" << macroBlockDim.x << "; " << macroBlockDim.y << ")" << std::endl;

	int filter_halfsize = 5;
	float sigma = float(filter_halfsize) / 2.0f;
	const float sigmaColor = 30.0f;

	const int runs = 2000;
	const int runs_prof = 1;
	const int warmUpRuns = 200;

	const std::vector<image::vec2ui> blockDims {
		{ 32, 8  },
		{ 16, 16 },
		{ 64, 4  },
		{ 32, 4  },
		{ 16, 8  },
		{ 8,  8  },

		// Fused Gaussian
		{ 16, 24 },
		{ 16, 32 },
		{ 32, 16 },
		{ 32, 24 },

		// Bilateral
		{ 8, 16  },
		{ 8, 32  }
	};

	{
		{
			const auto p = cuda::motion::BlockMatchingParams {
				{ 16, 16 },
				macroBlockDim,
				search_halfsize
			};

			for (int i = 0; i < warmUpRuns; ++i) {
				cuda::motion::BlockMatching(prevView, currView, outView, p, ctx);
			}
		}


		cudaStreamSynchronize(stream);
		profiler->Clear();


		//{
		//	// Cuda profiler
		//	const auto p = cuda::motion::BlockMatchingParams{
		//		{ 8, 8 },
		//		macroBlockDim,
		//		search_halfsize
		//	};

		//	for (int i = 0; i < runs_prof; ++i) {
		//		cuda::motion::BlockMatchingSimple(prevView, currView, outView, p, ctx);
		//	}

		//	cudaStreamSynchronize(stream);
		//	profiler->Clear();
		//}

		{
			// Perf test
			//{
			//	const auto p = cuda::motion::BlockMatchingParams {
			//		{ 8, 8 },
			//		macroBlockDim,
			//		search_halfsize
			//	};

			//	for (int i = 0; i < runs; ++i) {
			//		cuda::motion::BlockMatchingSimpleT(prevView, currView, outView, p, ctx);
			//	}
			//}


			for (const auto& blockDim : blockDims) {
				const auto p = cuda::motion::BlockMatchingParams {
					blockDim,
					macroBlockDim,
					search_halfsize
				};

				for (int i = 0; i < runs; ++i) {
					cuda::motion::BlockMatching(prevView, currView, outView, p, ctx);
				}
			}
		}
	}

	cudaCheck(cudaStreamSynchronize(stream));
	print::PrintKernelStats(profiler->GetRuns());

	//const auto img_motion = motion::BlockMatchingWarp(prevFrame, currFrame, { 16, 16 }, { 3, 3 }, { 16, 16 }, ctx);
	//const auto img_motion_cpu = image::ImageGpuToCpu(img_motion, stream);

	//const auto img_motion_s = motion::BlockMatchingSimple(prevFrame, currFrame, { 16, 16 }, { 3, 3 }, { 16, 16 }, ctx);
	//const auto img_motion_cpu_s = image::ImageGpuToCpu(img_motion_s, stream);

	//cudaCheck(cudaStreamSynchronize(stream));

	//{
	//	const auto* begin1 = img_motion_cpu.Data();
	//	const auto* end1 = begin1 + (img_motion_cpu.Dim().x * img_motion_cpu.Dim().y);

	//	const auto* begin2 = img_motion_cpu_s.Data();
	//	const auto* end2 = begin2 + (img_motion_cpu_s.Dim().x * img_motion_cpu_s.Dim().y);

	//	const size_t size = img_motion_cpu_s.Dim().x * img_motion_cpu_s.Dim().y;
	//	for (size_t i = 0; i < size; ++i) {
	//		auto val1 = *begin1;
	//		auto val2 = *begin2;

	//		if (val1.x != val2.x || val1.y != val2.y) {
	//			int a = 5;
	//			a = 4;
	//		}

	//		++begin1;
	//		++begin2;
	//	}
	//}

	//Image::WriteToFile(img_gauss_cpu, "C:\\AY\\Code\\ImgTest\\1_out_fused_v2.png");

	cudaCheck(cudaStreamDestroy(stream));

	return 0;
}
