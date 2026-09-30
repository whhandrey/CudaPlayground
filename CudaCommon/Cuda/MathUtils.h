#pragma once
#include <Image/ImageTypes.h>
#include <cuda_runtime.h>

namespace cuda {
	namespace math {
		template <class IntT>
		inline IntT DivUp(IntT x, IntT block1dDim) {
			return (x + block1dDim - 1) / block1dDim;
		}

		inline dim3 DivUp(image::vec2ui dim, image::vec2ui blockDim) {
			return {
				DivUp(dim.x, blockDim.x),
				DivUp(dim.y, blockDim.y),
				1
			};
		}

		inline dim3 DivUp(image::vec3ui dim, image::vec3ui blockDim) {
			return {
				DivUp(dim.x, blockDim.x),
				DivUp(dim.y, blockDim.y),
				DivUp(dim.z, blockDim.z)
			};
		}

		inline dim3 Div(image::vec2ui dim, image::vec2ui blockDim) {
			return {
				dim.x / blockDim.x,
				dim.y / blockDim.y,
				1
			};
		}

		inline dim3 vecTodim3(image::vec2ui dim) {
			return { dim.x, dim.y, 1 };
		}

		inline dim3 vecTodim3(image::vec3ui dim) {
			return { dim.x, dim.y, dim.z };
		}
	}
}
