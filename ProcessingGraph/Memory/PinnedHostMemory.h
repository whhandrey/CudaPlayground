#pragma once
#include <cstddef>

namespace cuda::memory {
    class PinnedHostMemory {
    public:
        PinnedHostMemory() = default;
        explicit PinnedHostMemory(std::size_t sizeBytes);

        ~PinnedHostMemory();

        PinnedHostMemory(const PinnedHostMemory&) = delete;
        PinnedHostMemory& operator=(const PinnedHostMemory&) = delete;

        PinnedHostMemory(PinnedHostMemory&& other) noexcept;
        PinnedHostMemory& operator=(PinnedHostMemory&& other) noexcept;

        void Allocate(std::size_t sizeBytes);
        void Release() noexcept;

        void* Data() const;
        std::size_t SizeBytes() const;

    private:
        void* m_data = nullptr;
        std::size_t m_sizeBytes = 0;
    };
}
