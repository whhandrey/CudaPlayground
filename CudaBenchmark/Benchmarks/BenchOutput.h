#pragma once
#include <vector>
#include <iostream>
#include <Cuda/Profiler/GpuProfileResult.h>

namespace bench::output {
	using cuda::profile::GpuProfileResult;

	void PrintCudaDevice();
	void PrintSharedMemStats();

	void PrintKernelStats(const std::vector<GpuProfileResult>& runs, std::ostream& os = std::cout);

	template <class Vec2d>
	std::string FormatVec2d(Vec2d dim) {
		return '(' +
			std::to_string(dim.x) + "x" +
			std::to_string(dim.y) + ')';
	}

	template <class Vec3d>
	std::string FormatVec3d(Vec3d dim) {
		return '(' +
			std::to_string(dim.x) + "x" +
			std::to_string(dim.y) + "x" +
			std::to_string(dim.z) + ')';
	}
}
