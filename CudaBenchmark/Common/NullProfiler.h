#pragma once
#include <Cuda/Profiler/IGpuProfiler.h>

namespace bench {
	using cuda::profile::ProfileSampleId;

	class NullProfiler : public cuda::profile::IGpuProfiler {
	public:
		ProfileSampleId BeginSample(const std::string& /*kernelName*/, cudaStream_t /*stream*/) { return ProfileSampleId{}; }
		void EndSample(ProfileSampleId /*id*/, cudaStream_t /*stream*/) {}
		void CancelSample(ProfileSampleId /*id*/) {}
	};
}
