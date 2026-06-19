#pragma once
#include <Image/ImageTypes.h>
#include <cuda_runtime.h>

namespace cuda {
	namespace math {
		inline dim3 DivUp(image::vec2ui dim, image::vec2ui blockSize) {
			return {
				(dim.x + blockSize.x - 1) / blockSize.x,
				(dim.y + blockSize.y - 1) / blockSize.y,
				1
			};
		}

		inline dim3 Div(image::vec2ui dim, image::vec2ui blockSize) {
			return {
				dim.x / blockSize.x,
				dim.y / blockSize.y,
				1
			};
		}

		inline dim3 vec2Todim3(image::vec2ui dim) {
			return { dim.x, dim.y, 1 };
		}
	}
}
