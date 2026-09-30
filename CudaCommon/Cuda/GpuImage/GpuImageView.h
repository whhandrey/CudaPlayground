#pragma once
#include <Image/ImageView.h>
#include "GpuImage.h"

namespace cuda::gpu_image {
	template <class T>
	image::GpuImageView<T> MakeImageView(GpuImage<T>& img) {
		return { img.Data(), img.Dim(), img.Pitch() };
	}

	template <class T>
	image::GpuImageView<const T> MakeImageView(const GpuImage<T>& img) {
		return { img.Data(), img.Dim(), img.Pitch() };
	}

	template <class T>
	image::GpuVolumeView<T> MakeVolumeView(GpuVolume<T>& vol) {
		return {
			vol.Data(),
			vol.Dim(),
			vol.Pitch(),
			vol.Pitch() * vol.Dim().y
		};
	}

	template <class T>
	image::GpuVolumeView<const T> MakeVolumeView(const GpuVolume<T>& vol) {
		return {
			vol.Data(),
			vol.Dim(),
			vol.Pitch(),
			vol.Pitch() * vol.Dim().y
		};
	}

	template <class T>
	image::GpuImageView<T> MakeEmptyImageView() {
		return { nullptr, {}, 0 };
	}

	template <class T>
	image::GpuVolumeView<T> MakeEmptyVolumeView() {
		return { nullptr, {}, 0, 0 };
	}
}
