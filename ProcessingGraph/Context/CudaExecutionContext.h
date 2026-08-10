#pragma once
#include <cuda_runtime.h>
#include <Profiler/IProfiler.h>

namespace dataflow {
	struct CudaExecutionContext {
		cudaStream_t stream;
		cuda::profile::IProfiler* profiler;
	};
}
