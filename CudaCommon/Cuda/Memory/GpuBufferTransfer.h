#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Memory/GpuSpan.h>

#include "GpuBuffer.h"
#include "GpuBufferSpan.h"

#include "../CudaCheck.h"

namespace cuda::gpu_buffer {
    template<class T>
    void Upload(const std::vector<T>& cpuInput, memory::GpuSpan<T> gpuOutput, cudaStream_t stream) {
        static_assert(!std::is_const_v<T>, "Upload destination must be writable");

        if (cpuInput.size() != gpuOutput.m_size) {
            throw std::logic_error("GpuBuffer::Upload: input/output size mismatch");
        }

        if (cpuInput.empty()) {
            return;
        }

        cudaCheck(cudaMemcpyAsync(
            gpuOutput.m_ptr,
            cpuInput.data(),
            cpuInput.size() * sizeof(T),
            cudaMemcpyHostToDevice,
            stream));
    }

    template<class T>
    GpuBuffer<T> Upload(const std::vector<T>& cpuInput, cudaStream_t stream) {
        GpuBuffer<T> gpuOutput(cpuInput.size());
        Upload(cpuInput, MakeSpan(gpuOutput), stream);

        return gpuOutput;
    }

    template<class T>
    void Download(memory::GpuSpan<const T> gpuInput, std::vector<T>& cpuOutput, cudaStream_t stream) {
        cpuOutput.resize(gpuInput.m_size);

        if (cpuOutput.empty()) {
            return;
        }

        cudaCheck(cudaMemcpyAsync(
            cpuOutput.data(),
            gpuInput.m_ptr,
            gpuInput.m_size * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream));
    }

    template<class T>
    std::vector<T> Download(memory::GpuSpan<const T> gpuInput, cudaStream_t stream) {
        std::vector<T> cpuOutput;
        Download(gpuInput,cpuOutput, stream);

        return cpuOutput;
    }
}