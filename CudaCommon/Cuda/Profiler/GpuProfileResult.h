#pragma once
#include <string>
#include <vector>

namespace cuda::profile {
	struct GpuProfileResult {
		std::string kernelName;
		std::vector<float> sampleDurationsMs;
	};
}
