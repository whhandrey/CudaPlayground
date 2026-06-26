#pragma once
#include <Image/ImageTypes.h>

namespace cuda {
	namespace motion {
		struct BlockMatchingParams {
			image::vec2ui blockDim;
			image::vec2ui macroBlockDim;
			image::vec2i search_halfsize;
		};

		// Same for dxdyOutput and confOutput
		image::vec2ui MotionOutputDim(image::vec2ui dim, image::vec2ui macroBlockDim);
	}
}
