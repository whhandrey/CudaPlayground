#include "../Image/Image.h"

namespace tile {
	namespace reduction {
		ImageGPU<float> Avg(const ImageGPU<uchar4>& input, ivec2 tileSize);
	}
}
