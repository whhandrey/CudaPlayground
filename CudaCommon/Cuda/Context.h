#pragma once
#include <cuda_runtime.h>
#include <memory>
#include "../Profiler/IProfiler.h"

namespace cuda {
	struct KernelContext {
		cudaStream_t m_stream;
		profile::IProfiler* m_profiler;
	};
}
