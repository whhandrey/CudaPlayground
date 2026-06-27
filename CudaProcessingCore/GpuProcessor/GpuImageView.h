#pragma once
#include <Image/ImageView.h>
#include "../DeviceImage/ImageGPU.h"

namespace cuda::image_view {
	template <class T>
	image::GpuImageView<T> MakeImageView(const ImageGPU<T>& img) {
		return { img.Data(), img.Dim(), img.Pitch() };
	}

	template <class T>
	image::GpuImageView<T> MakeEmptyImageView() {
		return { nullptr, {}, 0 };
	}
}
