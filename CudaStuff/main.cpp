#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Image/Image.h"
#include "Cuda/Filter.cuh"
#include "Cuda/Motion.cuh"
#include "Common/Logger.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>

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

	std::cout << "CUDA device: " << props.name << std::endl;
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

	const std::string file = "C:\\AY\\Code\\ImgTest\\1.jpg";
	
	ImageCPU<uchar4> img = Image::LoadFromFile<uchar4>(file);

	ImageGPU<uchar4> prevFrame(img, stream);
	ImageGPU<uchar4> currFrame(img, stream);

	currFrame = motion::shift::ShiftImage(currFrame, { -3, 2 }, stream);
	
	PrintCudaDevice();
	PrintSharedMemStats();

	std::cout << std::endl << "Testing image of (" << img.Dim().x << "; " << img.Dim().y << ")" << std::endl;

	int2 search_halfsize = { 3, 3 };
	ivec2 macroBlockDim = { 16, 16 };

	std::cout 
		<< "search_halfsize: (" << search_halfsize.x << "; " <<search_halfsize.y << ")\t" 
		<< "macroBlockDim: (" << macroBlockDim.x << "; " << macroBlockDim.y << ")" << std::endl;

	int filter_halfsize = 5;
	float sigma = float(filter_halfsize) / 2.0f;
	const float sigmaColor = 30.0f;

	const int runs = 1000;
	const int runs_prof = 1;
	const int warmUpRuns = 20;

	const std::vector<ivec2> blockDims {
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
		for (int i = 0; i < warmUpRuns; ++i) {
			const auto _ = motion::BlockMatchingSimpleT(prevFrame, currFrame, macroBlockDim, search_halfsize, { 16, 16 }, stream);
		}

		cudaStreamSynchronize(stream);
		cuda::Logger().Clear();


		//{
		//	// Cuda profiler
		//	for (int i = 0; i < runs_prof; ++i) {
		//		const auto _ = motion::BlockMatching(prevFrame, currFrame, macroBlockDim, search_halfsize, { 16, 16 }, stream);
		//	}

		//	cudaStreamSynchronize(stream);
		//	cuda::Logger().Clear();
		//}

		{
			// Perf test
			for (int i = 0; i < runs; ++i) {
				const auto _ = motion::BlockMatchingSimpleT(prevFrame, currFrame, macroBlockDim, search_halfsize, { 8, 8 }, stream);
			}

			//for (const auto& blockDim : blockDims) {
			//	for (int i = 0; i < runs; ++i) {
			//		const auto _ = motion::BlockMatchingSimpleT(prevFrame, currFrame, macroBlockDim, search_halfsize, blockDim, stream);
			//	}
			//}
		}
	}

	cudaCheck(cudaStreamSynchronize(stream));
	print::PrintKernelStats(cuda::Logger().GetRuns());

	const auto img_motion = motion::BlockMatchingWarp(prevFrame, currFrame, { 16, 16 }, { 3, 3 }, { 16, 16 }, stream);
	const auto img_motion_cpu = Image::ImageGpuToCpu(img_motion, stream);

	const auto img_motion_s = motion::BlockMatchingSimple(prevFrame, currFrame, { 16, 16 }, { 3, 3 }, { 16, 16 }, stream);
	const auto img_motion_cpu_s = Image::ImageGpuToCpu(img_motion_s, stream);

	cudaCheck(cudaStreamSynchronize(stream));

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


//int main() {
//	cudaStream_t stream;
//	cudaCheck(cudaStreamCreate(&stream));
//
//	std::vector<std::string> files{ "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg" };
//	const std::string filePath = "C:\\AY\\Code\\ImgTest\\";
//
//	std::ofstream fileStats(filePath + "stats.txt", std::ios::app);
//
//	for (const auto& fileName : files) {
//		const auto file = filePath + fileName;
//
//		ImageCPU<uchar4> img = Image::LoadFromFile<uchar4>(file);
//		ImageGPU<uchar4> img_gpu(img, stream);
//
//		fileStats << "Testing image of (" << img.Dim().x << "; " << img.Dim().y << ")" << std::endl;
//
//		int filter_halfsize = 5;
//		float sigma = float(filter_halfsize) / 2.0f;
//		const float sigmaColor = 30.0f;
//
//		const int runs = 1000;
//		const int warmUpRuns = 20;
//
//		const std::vector<ivec2> blockDims {
//			{ 32, 8  },
//			{ 16, 16 },
//			{ 64, 4  },
//			{ 32, 4  },
//			{ 16, 8  },
//
//			// Fused Gaussian
//			{ 16, 24 },
//			{ 16, 32 },
//			{ 32, 16 },
//			{ 32, 24 },
//
//			// Bilateral
//			{ 8, 16  },
//			{ 8, 32  }
//		};
//
//		{
//			for (int i = 0; i < warmUpRuns; ++i) {
//				const auto _ = filter::GaussianBlur(img_gpu, filter_halfsize, sigma, { 16, 16 }, stream);
//			}
//
//			cudaStreamSynchronize(stream);
//			cuda::Logger().Clear();
//
//			for (const auto& blockDim : blockDims) {
//				for (int i = 0; i < runs; ++i) {
//					const auto _ = filter::GaussianBlur(img_gpu, filter_halfsize, sigma, blockDim, stream);
//				}
//			}
//		}
//
//		cudaCheck(cudaStreamSynchronize(stream));
//		print::PrintKernelStats(cuda::Logger().GetRuns(), fileStats);
//
//		cuda::Logger().Clear();
//
//		{
//			for (int i = 0; i < warmUpRuns; ++i) {
//				const auto _ = filter::GaussianBlurFused(img_gpu, filter_halfsize, sigma, { 16, 16 }, stream);
//			}
//
//			cudaCheck(cudaStreamSynchronize(stream));
//			cuda::Logger().Clear();
//
//			for (const auto& blockDim : blockDims) {
//				for (int i = 0; i < runs; ++i) {
//					const auto _ = filter::GaussianBlurFused(img_gpu, filter_halfsize, sigma, blockDim, stream);
//				}
//			}
//		}
//
//		cudaCheck(cudaStreamSynchronize(stream));
//		print::PrintKernelStats(cuda::Logger().GetRuns(), fileStats);
//
//		cuda::Logger().Clear();
//
//		{
//			for (int i = 0; i < warmUpRuns; ++i) {
//				const auto _ = filter::BilateralFilter(img_gpu, filter_halfsize, sigma, sigmaColor, { 16, 16 }, stream);
//			}
//
//			cudaCheck(cudaStreamSynchronize(stream));
//			cuda::Logger().Clear();
//
//			for (const auto& blockDim : blockDims) {
//				for (int i = 0; i < runs; ++i) {
//					const auto _ = filter::BilateralFilter(img_gpu, filter_halfsize, sigma, sigmaColor, blockDim, stream);
//				}
//			}
//		}
//
//		cudaCheck(cudaStreamSynchronize(stream));
//		print::PrintKernelStats(cuda::Logger().GetRuns(), fileStats);
//
//		cuda::Logger().Clear();
//
//		//const auto img_filt_gpu_sep = filter::GaussianBlur(img_gpu, filter_halfsize, sigma, stream);
//		//const auto img_filt_cpu_sep = Image::ImageGpuToCpu(img_filt_gpu_sep, stream);
//
//		//const auto img_filt_gpu_fused = filter::GaussianBlurFused(img_gpu, filter_halfsize, sigma, stream);
//		//const auto img_filt_cpu_fused = Image::ImageGpuToCpu(img_filt_gpu_fused, stream);
//
//		//const float sigmaColor = 30.0f;
//		//const auto img_bilateral = filter::BilateralFilter(img_gpu, filter_halfsize, sigma, sigmaColor, stream);
//		//const auto img_bilateral_cpu = Image::ImageGpuToCpu(img_bilateral, stream);
//
//		//cudaCheck(cudaStreamSynchronize(stream));
//
//		//Image::WriteToFile(img_filt_cpu_sep, "C:\\AY\\Code\\ImgTest\\1_out_sep.png");
//		//Image::WriteToFile(img_filt_cpu_fused, "C:\\AY\\Code\\ImgTest\\1_out_fused.png");
//		//Image::WriteToFile(img_filt_cpu_fused, "C:\\AY\\Code\\ImgTest\\1_out_bilateral.png");
//	}
//
//	cudaCheck(cudaStreamDestroy(stream));
//
//	return 0;
//}
