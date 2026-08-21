#pragma once
#include <cuda_runtime.h>
#include <memory>
#include "Profiler/IGpuProfiler.h"

namespace cuda {
	struct KernelContext {
		cudaStream_t stream;
		profile::IGpuProfiler* profiler;
	};
}
