#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include "GpuImageView.h"

namespace cuda::gpu_image {
	template <class DstSample, class SrcSample>
	image::Image<DstSample> DownloadCompatible(
		image::GpuImageView<SrcSample> img,
		cudaStream_t stream)
	{
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for download");

		image::Image<DstSample> output(img.m_dim);

		cudaCheck(cudaMemcpy2DAsync(
			output.Data(),
			output.Dim().x * sizeof(DstSample),
			img.m_ptr,
			img.m_pitch,
			img.m_dim.x * sizeof(SrcSample),
			img.m_dim.y,
			cudaMemcpyDeviceToHost,
			stream
		));

		return output;
	}

	template <class T>
	inline image::Image<T> Download(const image::GpuImageView<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(img, stream);
	}

	template <class T>
	inline image::Image<T> Download(const ImageGPU<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(cuda::gpu_image::MakeImageView(img), stream);
	}

	template <class DstSample, class SrcSample>
	void UploadCompatible(
		const image::ImageView<SrcSample>& in_cpu,
		image::GpuImageView<DstSample> out_gpu,
		cudaStream_t stream)
	{
		static_assert(sizeof(DstSample) == sizeof(SrcSample),"Src/Dst sample sizes must match for upload");

		if (in_cpu.m_dim.x != out_gpu.m_dim.x || in_cpu.m_dim.y != out_gpu.m_dim.y) {
			throw std::logic_error("UploadCompatible: trying to upload an image with wrong dim");
		}

		cudaCheck(cudaMemcpy2DAsync(
			out_gpu.m_ptr,
			out_gpu.m_pitch,
			in_cpu.m_ptr,
			in_cpu.m_pitch,
			in_cpu.m_dim.x * sizeof(SrcSample),
			in_cpu.m_dim.y,
			cudaMemcpyHostToDevice,
			stream
		));
	}

	template <class DstSample, class SrcSample>
	void UploadCompatible(
		const image::ImageView<SrcSample>& in_cpu,
		cuda::ImageGPU<DstSample>& out_gpu,
		cudaStream_t stream)
	{
		UploadCompatible<DstSample>(in_cpu, cuda::gpu_image::MakeImageView(out_gpu), stream);
	}

	template <class T>
	void Upload(const image::ImageView<T>& in_cpu, image::GpuImageView<T> out_gpu, cudaStream_t stream) {
		UploadCompatible<T>(in_cpu, out_gpu, stream);
	}

	template <class T>
	ImageGPU<T> Create(const image::ImageView<T>& in_cpu, cudaStream_t stream) {
		ImageGPU<T> out(in_cpu.m_dim);
		Upload(in_cpu, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}

	template <class T>
	ImageGPU<T> Create(const image::Image<T>& in_cpu, cudaStream_t stream) {
		ImageGPU<T> out(in_cpu.m_dim);

		auto in_view = image::ImageView<T>{ in_cpu.Data(), in_cpu.Dim(), size_t(in_cpu.Dim().x * sizeof(T)) };
		Upload(in_view, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}
}
