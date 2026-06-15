#pragma once
#include "Common.h"

class CudaTimer {
public:
    explicit CudaTimer(cudaStream_t stream = 0)
        : m_stream(stream)
    {
        cudaCheck(cudaEventCreate(&m_start));
        cudaCheck(cudaEventCreate(&m_stop));

        Begin();
    }

    ~CudaTimer()
    {
        cudaCheck(cudaEventDestroy(m_start));
        cudaCheck(cudaEventDestroy(m_stop));
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

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
