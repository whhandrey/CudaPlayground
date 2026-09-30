#include "RandomData.h"
#include <cstdint>
#include <random>

namespace bench::data {
	image::CpuVolume<float> GenerateRandomVolume(image::vec3ui dim, std::uint32_t seed) {
		image::CpuVolume<float> volume(dim);
		const std::size_t sampleCount = dim.x * dim.y * dim.z;

		std::mt19937 gen(seed);
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

		float* data = volume.Data();

		for (std::size_t i = 0; i < sampleCount; ++i) {
			data[i] = distribution(gen);
		}

		return volume;
	}

	std::vector<float> GenerateRandomWeights3d(image::vec3i filter_halfsize, std::uint32_t seed) {
		const std::size_t width = filter_halfsize.x * 2 + 1;
		const std::size_t height = filter_halfsize.y * 2 + 1;
		const std::size_t depth = filter_halfsize.z * 2 + 1;

		std::vector<float> weights(width * height * depth);

		std::mt19937 gen(seed);
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

		for (float& weight : weights) {
			weight = distribution(gen);
		}

		return weights;
	}

	std::vector<float> GenerateRandomWeights(unsigned int filter_halfsize, std::uint32_t seed) {
		const std::size_t length = filter_halfsize * 2 + 1;
		std::vector<float> weights(length);

		std::mt19937 gen(seed);
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

		for (float& weight : weights) {
			weight = distribution(gen);
		}

		return weights;
	}
}
