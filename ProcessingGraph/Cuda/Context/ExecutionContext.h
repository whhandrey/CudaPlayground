#pragma once
#include <cuda_runtime.h>
#include <Profiler/IProfiler.h>

namespace cuda::dataflow {
	struct ExecutionContext {
		cudaStream_t stream;
		profile::IProfiler* profiler;
	};
}
