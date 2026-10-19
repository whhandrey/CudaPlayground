#pragma once

#include <Image/ImageView.h>
#include <Memory/GpuSpan.h>
#include <Cuda/KernelContext.h>
#include "LocalMinBlending.h"

namespace cuda {
	namespace synth_vol_pipeline {
		struct Workspace {
			image::GpuVolumeView<float> tmp1;
			image::GpuVolumeView<float> tmp2;
			image::GpuVolumeView<float> filteredData;
		};

		void VolumePipelineRef(
			image::GpuVolumeView<float> inputOutput,
			memory::GpuSpan<const float> coeffs3d,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx
		);

		void VolumePipelineConv3dOptABit(
			image::GpuVolumeView<float> inputOutput,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx
		);

		image::GpuVolumeView<float> VolumePipelineOpt(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			erode3d::BlendParams blendParams,
			Workspace ws,
			image::vec3i conv3d_halfsize,
			image::vec3i erode3d_halfsize,
			cuda::KernelContext ctx
		);
	}
}
