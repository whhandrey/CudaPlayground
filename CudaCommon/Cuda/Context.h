#pragma once
#include <cuda_runtime.h>
#include <memory>
#include "Profiler.h"

namespace cuda {
	struct KernelContext {
		cudaStream_t m_stream;
		std::shared_ptr<profiler::IProfiler> m_profiler;
	};
}
