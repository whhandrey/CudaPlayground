#pragma once
#include <vector>

struct ImageViewRGBA8 {
	const unsigned char* data;
	int width = 0;
	int height = 0;
	int pitchBytes = 0;
};

struct ImageRGBA8 {
	std::vector<unsigned char> data;
	int width = 0;
	int height = 0;
	int pitchBytes = 0;
};

namespace gpu {
	class IMotionGpuProcessor {
	public:
		virtual ~IMotionGpuProcessor() = default;
		virtual ImageRGBA8 Process(const ImageViewRGBA8& prev, const ImageViewRGBA8& next) = 0;
	};
}
