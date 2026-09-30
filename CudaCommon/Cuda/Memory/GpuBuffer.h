#pragma once
#include <cstddef>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include "../CudaCheck.h"
#include <Memory/GpuSpan.h>

namespace cuda::gpu_buffer {
    template<class T>
    class GpuBuffer {
        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_void_v<T>);

    public:
        GpuBuffer() = default;

        explicit GpuBuffer(std::size_t size)
            : m_size(size)
        {
            if (m_size == 0) {
                return;
            }

            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&m_ptr), SizeBytes()));
        }

        ~GpuBuffer() {
            Destroy();
        }

        GpuBuffer(const GpuBuffer&) = delete;
        GpuBuffer& operator=(const GpuBuffer&) = delete;

        GpuBuffer(GpuBuffer&& other) noexcept
            : m_ptr(std::exchange(other.m_ptr, nullptr)),
            m_size(std::exchange(other.m_size, 0))
        {
        }

        GpuBuffer& operator=(GpuBuffer&& other) noexcept {
            if (this != &other) {
                Destroy();

                m_ptr = std::exchange(other.m_ptr, nullptr);
                m_size = std::exchange(other.m_size, 0);
            }

            return *this;
        }

        T* Data() {
            return m_ptr;
        }

        const T* Data() const {
            return m_ptr;
        }

        std::size_t Size() const {
            return m_size;
        }

        std::size_t SizeBytes() const {
            return m_size * sizeof(T);
        }

        bool Empty() const {
            return m_size == 0;
        }

    private:
        void Destroy() noexcept {
            if (m_ptr) {
                // Destructors must not throw.
                cudaFree(m_ptr);
            }

            m_ptr = nullptr;
            m_size = 0;
        }

    private:
        T* m_ptr = nullptr;
        std::size_t m_size = 0;
    };
}
