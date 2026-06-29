#pragma once
#include <Image/ImageView.h>
#include "../DeviceImage/ImageGPU.h"

namespace cuda::gpu_image {
	template <class T>
	image::GpuImageView<T> MakeImageView(ImageGPU<T>& img) {
		return { img.Data(), img.Dim(), img.Pitch() };
	}

	template <class T>
	image::GpuImageView<T> MakeEmptyImageView() {
		return { nullptr, {}, 0 };
	}
}
