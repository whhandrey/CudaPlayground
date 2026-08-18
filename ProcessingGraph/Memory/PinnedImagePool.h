#pragma once
#include "PinnedHostMemory.h"
#include <Image/ImageView.h>
#include <stdexcept>

namespace cuda::memory {
    inline std::size_t AlignUp(std::size_t value, std::size_t alignment) {
        // DivUp(value, alignment) * alignment
        return (value + alignment - 1) / alignment * alignment;
    }

    template <class SampleType>
    class PinnedImagePool {
    public:
        PinnedImagePool(image::vec2ui maxDim, std::size_t maxSlotCount)
            : m_maxDim{ maxDim }
            , m_maxSlotCount{ maxSlotCount }
            , m_slotPitch{ maxDim.x * sizeof(SampleType) }
            , m_slotSizeBytes{ m_slotPitch * maxDim.y }
        {
            const size_t maxMemSize = m_slotSizeBytes * m_maxSlotCount;
            m_memory = PinnedHostMemory(maxMemSize);
        }

        image::PinnedImageView<SampleType> View(image::vec2ui dim, std::size_t index) {
            if (index >= SlotCount()) {
                throw std::out_of_range("PinnedImagePool: invalid image index");
            }

            if (dim.x > m_maxDim.x || dim.y > m_maxDim.y) {
                throw std::logic_error("PinnedImagePool: exceeded max allowed dim");
            }

            // pitch is the same as dim.x for now
            const size_t pitchBytes = dim.x * sizeof(SampleType);

            auto* base = static_cast<std::byte*>(m_memory.Mem());
            auto* imageData = base + index * m_slotSizeBytes;

            return {
                reinterpret_cast<SampleType*>(imageData),
                dim,
                pitchBytes
            };
        }

        std::size_t SlotCount() const noexcept {
            return m_maxSlotCount;
        }

    private:
        const image::vec2ui m_maxDim;
        const size_t m_maxSlotCount;
        const size_t m_slotPitch;
        const size_t m_slotSizeBytes;

        PinnedHostMemory m_memory;
    };
}
