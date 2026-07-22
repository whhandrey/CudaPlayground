#include "Params.h"

namespace cuda::motion::visualization {
	image::vec2ui ArrowsMapOutputDim(image::vec2ui statsDim, image::vec2ui renderDim) {
		return {
			statsDim.x * renderDim.x,
			statsDim.y * renderDim.y
		};
	}
}
