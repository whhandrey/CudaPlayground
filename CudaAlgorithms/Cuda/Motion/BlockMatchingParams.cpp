#include "BlockMatchingParams.h"
#include <Cuda/MathUtils.h>
#include "../VecUtils.h"

namespace cuda {
	namespace motion {
		bool BlockMatchingParams::operator ==(const BlockMatchingParams& p) const {
			return util::vec::Eq(p.blockDim, blockDim)
				&& util::vec::Eq(p.macroBlockDim, macroBlockDim)
				&& util::vec::Eq(p.search_halfsize, search_halfsize);
		}

		image::vec2ui MotionOutputDim(image::vec2ui dim, image::vec2ui macroBlockDim) {
			auto outputDim3 = math::Div(dim, macroBlockDim);

			return {
				outputDim3.x,
				outputDim3.y
			};
		}
	}
}
