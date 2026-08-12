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
        PinnedImagePool(image::vec2ui dim, std::size_t imageCount)
            : m_dim{ dim }
            , m_imageCount{ imageCount }
            , m_pitchBytes{ static_cast<std::size_t>(dim.x) * sizeof(SampleType) }
            , m_imageSizeBytes{ m_pitchBytes * static_cast<std::size_t>(dim.y) }
        {
            constexpr size_t maxMemSize = MaxDim.x * MaxDim.y * SlotCount * sizeof(SampleType);
            if (m_imageSizeBytes * m_imageCount > maxMemSize) {
                throw std::logic_error("PinnedImagePool: requested larger than supported image dim/count");
            }

            m_memory = PinnedHostMemory(maxMemSize);
        }

        image::PinnedImageView<SampleType> View(std::size_t index) {
            if (index >= m_imageCount) {
                throw std::out_of_range("PinnedImagePool: invalid image index");
            }

            auto* base = static_cast<std::byte*>(m_memory.Data());
            auto* imageData = base + index * m_imageSizeBytes;

            return {
                reinterpret_cast<SampleType*>(imageData),
                m_dim,
                m_pitchBytes
            };
        }

        std::size_t ImageCount() const noexcept {
            return m_imageCount;
        }

        std::size_t ImageSizeBytes() const noexcept {
            return m_imageSizeBytes;
        }

    private:
        static constexpr image::vec2ui MaxDim{ 3840, 2160 };
        static constexpr std::size_t SlotCount = 5;

        image::vec2ui m_dim{};

        std::size_t m_imageCount = 0;
        std::size_t m_pitchBytes = 0;
        std::size_t m_imageSizeBytes = 0;

        PinnedHostMemory m_memory;
    };
}
