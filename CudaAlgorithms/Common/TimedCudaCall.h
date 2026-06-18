#pragma once
#include "Common.h"
#include "../Context/Context.h"

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
	void TimedCall(const std::string& name, cuda::KernelContext& ctx, Fn&& cudaKernel) {

		{
			Timer timer(ctx.m_stream);
			
			cudaKernel();

            if (cudaGetLastError() == cudaSuccess) {
                ctx.m_profiler->Profile(name, timer.EndMs());
            }
		}
	}
}
