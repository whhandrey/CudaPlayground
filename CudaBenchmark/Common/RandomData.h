#pragma once
#include <Image/Image.h>

namespace bench::data {
    image::CpuVolume<float> GenerateRandomVolume(image::vec3ui dim, std::uint32_t seed = 42);

    // full convolution kinda
    std::vector<float> GenerateRandomWeights3d(image::vec3i filter_halfsize, std::uint32_t seed = 1337);

    std::vector<float> GenerateRandomWeights(unsigned int filter_halfsize, std::uint32_t seed = 1337);
}
