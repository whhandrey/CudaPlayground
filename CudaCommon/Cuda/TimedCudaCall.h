#pragma once
#include "KernelContext.h"
#include "CudaCheck.h"

namespace cuda {
    class Timer {
    public:
        explicit Timer(cudaStream_t stream = 0)
            : m_stream(stream)
        {
            cudaCheck(cudaEventCreate(&m_start));
            cudaCheck(cudaEventCreate(&m_stop));

            Begin();
        }

        ~Timer()
        {
            cudaCheck(cudaEventDestroy(m_start));
            cudaCheck(cudaEventDestroy(m_stop));
        }

        Timer(const Timer&) = delete;
        Timer& operator=(const Timer&) = delete;

        inline float EndMs() {
            cudaEventRecord(m_stop, m_stream);
            cudaEventSynchronize(m_stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, m_start, m_stop);
            return ms;
        }

    private:
        inline void Begin() {
            cudaEventRecord(m_start, m_stream);
        }

    private:
        cudaEvent_t m_start{};
        cudaEvent_t m_stop{};
        cudaStream_t m_stream{};
    };

	template <class Fn>
	void TimedCall(const std::string& kernelName, cuda::KernelContext& ctx, Fn&& cudaKernel) {
        const auto id = ctx.profiler->BeginSample(kernelName, ctx.stream);
			
        std::forward<Fn>(cudaKernel)();

        // Ignore errors for now, needed to test different blockDim sizes which might be invalid in some cases.
        if (cudaGetLastError() == cudaSuccess) {
            ctx.profiler->EndSample(id, ctx.stream);
        }
        else {
            ctx.profiler->CancelSample(id);
        }
	}
}
