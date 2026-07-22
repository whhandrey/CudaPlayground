#pragma once
#include <Image/ImageTypes.h>

namespace cuda::motion::visualization {
	image::vec2ui ArrowsMapOutputDim(image::vec2ui statsDim, image::vec2ui renderDim);
}
