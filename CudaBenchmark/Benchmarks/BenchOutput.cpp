#include "BenchOutput.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <Cuda/CudaCheck.h>
#include <Image/ImageTypes.h>

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

namespace bench::output {
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

	void PrintKernelStats(const std::vector<GpuProfileResult>& runs, std::ostream& os) {
		os << std::fixed << std::setprecision(4);

		os << "\nCUDA kernel timings:\n\n";

		int kernelW = 0;
		for (const auto& [scope, name, _] : runs) {
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

		for (const auto& [_, name, times] : runs) {
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
