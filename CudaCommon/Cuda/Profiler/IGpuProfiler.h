#pragma once
#include <string>
#include <cuda_runtime.h>

namespace cuda {
	namespace profile {
        using ProfileSampleId = std::size_t;

        class IGpuProfiler {
        public:
            virtual ~IGpuProfiler() = default;

            virtual ProfileSampleId BeginSample(const std::string& kernelName, cudaStream_t stream) = 0;
            virtual void EndSample(ProfileSampleId id, cudaStream_t stream) = 0;

            virtual void CancelSample(ProfileSampleId id) = 0;
        };
	}
}
