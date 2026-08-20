#pragma once
#include <cuda_runtime.h>
#include <Image/Image.h>
#include <Image/ImageView.h>
#include "GpuImageView.h"
#include "ImageGPU.h"
#include "../CudaCheck.h"

namespace cuda::gpu_image {
	template<class Dst, class Src>
	struct IsDownloadCompatibleImpl
		: std::bool_constant<std::is_same_v<Dst, Src>>
	{
	};

	template<>
	struct IsDownloadCompatibleImpl<image::vec4uc, uchar4>
		: std::true_type
	{
	};

	template<class Dst, class Src>
	struct IsDownloadCompatible : IsDownloadCompatibleImpl<std::remove_cvref_t<Dst>, std::remove_cvref_t<Src>>
	{
	};

	template <class DstSample, class SrcSample, template<class> class CpuView>
	void DownloadCompatible(
		const image::GpuImageView<SrcSample>& gpu_in,
		CpuView<DstSample> cpu_out,
		cudaStream_t stream)
	{
		static_assert(IsDownloadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for download");

		if (gpu_in.m_dim.x != cpu_out.m_dim.x || gpu_in.m_dim.y != cpu_out.m_dim.y) {
			throw std::logic_error("DownloadCompatible: trying to download an image with wrong dim");
		}

		const size_t rowBytes = gpu_in.m_dim.x * sizeof(SrcSample);
		if (gpu_in.m_pitch < rowBytes || cpu_out.m_pitch < rowBytes) {
			throw std::logic_error("DownloadCompatible: pitch is smaller than row size");
		}

		cudaCheck(cudaMemcpy2DAsync(
			cpu_out.m_ptr,
			cpu_out.m_dim.x * sizeof(DstSample),
			gpu_in.m_ptr,
			gpu_in.m_pitch,
			gpu_in.m_dim.x * sizeof(SrcSample),
			gpu_in.m_dim.y,
			cudaMemcpyDeviceToHost,
			stream
		));
	}

	template <class DstSample, class SrcSample>
	image::Image<DstSample> DownloadCompatible(
		const image::GpuImageView<SrcSample>& gpu_in,
		cudaStream_t stream)
	{
		image::Image<DstSample> cpu_out(gpu_in.m_dim);
		const auto cpu_view_out = image::CpuImageView<DstSample>{ cpu_out.Data(), cpu_out.Dim(), cpu_out.Pitch() };

		DownloadCompatible<DstSample>(gpu_in, cpu_view_out, stream);

		return cpu_out;
	}

	template <class T>
	inline image::Image<T> Download(const image::GpuImageView<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(img, stream);
	}

	template <class T>
	inline image::Image<T> Download(const ImageGPU<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(cuda::gpu_image::MakeImageView(img), stream);
	}

	template<class Dst, class Src>
	struct IsUploadCompatibleImpl
		: std::bool_constant<std::is_same_v<Dst, Src>>
	{
	};

	template<>
	struct IsUploadCompatibleImpl<uchar4, image::vec4uc>
		: std::true_type
	{
	};

	template<class Dst, class Src>
	struct IsUploadCompatible : IsUploadCompatibleImpl<std::remove_cvref_t<Dst>, std::remove_cvref_t<Src>>
	{
	};

	template <class DstSample, class SrcSample, template<class> class CpuView>
	void UploadCompatible(
		const CpuView<SrcSample>& in_cpu,
		image::GpuImageView<DstSample> out_gpu,
		cudaStream_t stream)
	{
		static_assert(IsUploadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for upload");

		if (in_cpu.m_dim.x != out_gpu.m_dim.x || in_cpu.m_dim.y != out_gpu.m_dim.y) {
			throw std::logic_error("UploadCompatible: trying to upload an image with wrong dim");
		}

		const size_t rowBytes = in_cpu.m_dim.x * sizeof(SrcSample);
		if (in_cpu.m_pitch < rowBytes || out_gpu.m_pitch < rowBytes) {
			throw std::logic_error("UploadCompatible: pitch is smaller than row size");
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
		const image::CpuImageView<SrcSample>& in_cpu,
		ImageGPU<DstSample>& out_gpu,
		cudaStream_t stream)
	{
		UploadCompatible<DstSample>(in_cpu, cuda::gpu_image::MakeImageView(out_gpu), stream);
	}

	template <class T>
	void Upload(const image::CpuImageView<const T>& in_cpu, image::GpuImageView<T> out_gpu, cudaStream_t stream) {
		UploadCompatible<T>(in_cpu, out_gpu, stream);
	}

	template <class T>
	ImageGPU<T> Create(const image::CpuImageView<T>& in_cpu, cudaStream_t stream) {
		ImageGPU<T> out(in_cpu.m_dim);
		Upload(in_cpu, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}

	template <class T>
	ImageGPU<T> Create(const image::Image<T>& in_cpu, cudaStream_t stream) {
		ImageGPU<T> out(in_cpu.m_dim);

		auto in_view = image::CpuImageView<const T>{ in_cpu.Data(), in_cpu.Dim(), size_t(in_cpu.Dim().x * sizeof(T)) };
		Upload(in_view, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}
}
