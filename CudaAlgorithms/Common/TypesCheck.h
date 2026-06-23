#pragma once
#include <Image/ImageTypes.h>
#include <type_traits>
#include <vector_types.h>

#define CHECK_SAME_LAYOUT(my_t, cuda_t)                           \
    static_assert(sizeof(my_t) == sizeof(cuda_t));                \
    static_assert(alignof(my_t) == alignof(cuda_t));              \
    static_assert(offsetof(my_t, x) == offsetof(cuda_t, x));      \
    static_assert(offsetof(my_t, y) == offsetof(cuda_t, y))

#define CHECK_SAME_LAYOUT4(my_t, cuda_t)                          \
    CHECK_SAME_LAYOUT(my_t, cuda_t);                              \
    static_assert(offsetof(my_t, z) == offsetof(cuda_t, z));      \
    static_assert(offsetof(my_t, w) == offsetof(cuda_t, w))

namespace cuda {
    namespace layout_check {
        CHECK_SAME_LAYOUT4(image::vec4uc, uchar4);

        CHECK_SAME_LAYOUT(image::vec2ui, uint2);
        CHECK_SAME_LAYOUT(image::vec2i, int2);

        CHECK_SAME_LAYOUT4(image::vec4ui, uint4);
        CHECK_SAME_LAYOUT4(image::vec4i, int4);
        CHECK_SAME_LAYOUT4(image::vec4f, float4);
    }
}
