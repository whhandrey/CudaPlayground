#pragma once
#include <Image/Image.h>
#include <cuda_runtime.h>
#include "ImageGPU.h"
#include "GpuImageView.h"

namespace cuda::transfer {
	template <class T>
	inline image::Image<T> ImageGpuToCpu(const image::GpuImageView<T>& img, cudaStream_t stream) {
		image::Image<T> output(img.m_dim);

		cudaMemcpy2DAsync(
			output.Data(),
			output.Dim().x * sizeof(T),
			img.m_ptr,
			img.m_pitch,
			img.m_dim.x * sizeof(T),
			img.m_dim.y,
			cudaMemcpyDeviceToHost,
			stream
		);

		return output;
	}

	template <class T>
	inline image::Image<T> ImageGpuToCpu(const ImageGPU<T>& img, cudaStream_t stream) {
		return ImageGpuToCpu(cuda::image_view::MakeImageView(img), stream);
	}
}
