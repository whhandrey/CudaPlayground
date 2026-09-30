#pragma once
#include <Image/Image.h>

namespace bench::data {
    image::CpuVolume<float> GenerateRandomVolume(image::vec3ui dim, std::uint32_t seed = 42);
    std::vector<float> GenerateRandomWeights(image::vec3i filter_halfsize, std::uint32_t seed = 1337);
}
