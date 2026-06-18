#pragma once
#include <Image/Image.h>
#include <DeviceImage/ImageGPU.h>
#include <cuda_runtime.h>

namespace image {
	template <class T>
	inline image::Image<T> ImageGpuToCpu(const ImageGPU<T>& img, cudaStream_t stream) {
		image::Image<T> output(img.Dim());

		cudaMemcpy2DAsync(
			output.Data(),
			output.Dim().x * sizeof(T),
			img.Data(),
			img.Pitch(),
			img.Dim().x * sizeof(T),
			img.Dim().y,
			cudaMemcpyDeviceToHost,
			stream
		);

		return output;
	}
}
