#include "BlockMatchingParams.h"
#include <Cuda/MathUtils.h>

namespace cuda {
	namespace motion {
		image::vec2ui MotionOutputDim(image::vec2ui dim, image::vec2ui macroBlockDim) {
			auto outputDim3 = math::Div(dim, macroBlockDim);

			return {
				outputDim3.x,
				outputDim3.y
			};
		}
	}
}
