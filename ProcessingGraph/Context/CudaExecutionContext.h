#pragma once
#include <cuda_runtime.h>
#include <Cuda/Profiler/IGpuProfiler.h>

namespace dataflow {
	struct CudaExecutionContext {
		cudaStream_t stream;
		cuda::profile::IGpuProfiler* profiler;
	};
}
