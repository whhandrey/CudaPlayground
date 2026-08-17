#pragma once
#include "../ResourceDesc.h"
#include "UntypedResourceView.h"
#include <Image/ImageView.h>
#include <stdexcept>

namespace processing::resource {
    template <class View>
    View MakeResourceView(const UntypedResourceView& raw, MemoryDomain domain) {
        using SampleType = typename View::SampleType;

        if (raw.domain != domain) {
            throw std::logic_error("Invalid resource domain");
        }

        if (raw.sampleType != typeid(SampleType)) {
            throw std::logic_error("Invalid sample type");
        }

        return {
            static_cast<SampleType*>(raw.data),
            raw.dim,
            raw.pitch
        };
    }

    template <class View>
    struct ResourceViewTraits;

    template <class SampleType>
    struct ResourceViewTraits<image::GpuImageView<SampleType>> {

        static image::GpuImageView<SampleType> Make(const UntypedResourceView& raw) {
            return MakeResourceView<image::GpuImageView<SampleType>>(raw, MemoryDomain::CudaDevice);
        }
    };

    template <class SampleType>
    struct ResourceViewTraits<image::CpuImageView<SampleType>> {

        static image::CpuImageView<SampleType> Make(const UntypedResourceView& raw) {
            return MakeResourceView<image::CpuImageView<SampleType>>(raw, MemoryDomain::Host);
        }
    };

    template <class SampleType>
    struct ResourceViewTraits<image::PinnedImageView<SampleType>> {

        static image::PinnedImageView<SampleType> Make(const UntypedResourceView& raw) {
            return MakeResourceView<image::PinnedImageView<SampleType>>(raw, MemoryDomain::HostPinned);
        }
    };
}
