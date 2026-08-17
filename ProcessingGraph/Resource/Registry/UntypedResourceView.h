#pragma once
#include "../ResourceDesc.h"
#include <Image/ImageView.h>
#include <stdexcept>

namespace processing::resource {
    struct UntypedResourceView {
        void* data;
        image::vec2ui dim;
        std::size_t pitch;
        MemoryDomain domain;
        std::type_index sampleType{ typeid(void) };
    };
}
