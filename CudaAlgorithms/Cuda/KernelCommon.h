#pragma once
#include <Image/ImageTypes.h>
#include <string>
#include <vector>

namespace cuda {
	namespace util {
		inline std::string BlockDimToString(const image::vec2ui& blockDim) {
			return "(" + std::to_string(blockDim.x) + ", " + std::to_string(blockDim.y) + ")";
		}
	}

	namespace gpu {
		enum class Arch {
			Unknown,
			Turing,
			Ampere,
			Ada,
			Blackwell
		};

		Arch DetectArch(int device = 0);
	}
}
