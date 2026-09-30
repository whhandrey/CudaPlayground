#pragma once
#include <cuda_runtime.h>
#include <Image/Image.h>
#include <Image/ImageView.h>
#include "GpuImageView.h"
#include "GpuImage.h"
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
	image::CpuImage<DstSample> DownloadCompatible(
		const image::GpuImageView<SrcSample>& gpu_in,
		cudaStream_t stream)
	{
		image::CpuImage<DstSample> cpu_out(gpu_in.m_dim);
		const auto cpu_view_out = image::CpuImageView<DstSample>{ cpu_out.Data(), cpu_out.Dim(), cpu_out.Pitch() };

		DownloadCompatible<DstSample>(gpu_in, cpu_view_out, stream);

		return cpu_out;
	}

	template <class T>
	inline image::CpuImage<T> Download(const image::GpuImageView<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(img, stream);
	}

	template <class T>
	inline image::CpuImage<T> Download(const GpuImage<T>& img, cudaStream_t stream) {
		return DownloadCompatible<T>(cuda::gpu_image::MakeImageView(img), stream);
	}

	// 3d download
	template <class DstSample, class SrcSample, template<class> class CpuView>
	void DownloadCompatible(
		const image::GpuVolumeView<SrcSample>& gpu_in,
		CpuView<DstSample> cpu_out,
		cudaStream_t stream)
	{
		static_assert(IsDownloadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for download");

		if (gpu_in.m_dim.x != cpu_out.m_dim.x || gpu_in.m_dim.y != cpu_out.m_dim.y || gpu_in.m_dim.z != cpu_out.m_dim.z) {
			throw std::logic_error("DownloadCompatible: trying to download a volume with wrong dim");
		}

		const size_t rowBytes = gpu_in.m_dim.x * sizeof(SrcSample);
		if (gpu_in.m_pitch < rowBytes || cpu_out.m_pitch < rowBytes) {
			throw std::logic_error("DownloadCompatible: pitch is smaller than row size");
		}

		const size_t sliceBytes = rowBytes * gpu_in.m_dim.y;
		if (gpu_in.m_slicePitch < sliceBytes || cpu_out.m_slicePitch < sliceBytes) {
			throw std::logic_error("DownloadCompatible: slicePitch is smaller than rowBytes * dim.y");
		}

		cudaMemcpy3DParms params{};

		params.srcPtr = make_cudaPitchedPtr(gpu_in.m_ptr, gpu_in.m_pitch, gpu_in.m_dim.x * sizeof(SrcSample), gpu_in.m_dim.y);
		params.dstPtr = make_cudaPitchedPtr(cpu_out.m_ptr, cpu_out.m_pitch, cpu_out.m_dim.x * sizeof(DstSample), cpu_out.m_dim.y);

		params.extent = make_cudaExtent(gpu_in.m_dim.x * sizeof(SrcSample), gpu_in.m_dim.y, gpu_in.m_dim.z);
		params.kind = cudaMemcpyDeviceToHost;

		cudaCheck(cudaMemcpy3DAsync(&params, stream));
	}

	template <class DstSample, class SrcSample>
	image::CpuImage<DstSample> DownloadCompatible(
		const image::GpuVolumeView<SrcSample>& gpu_in,
		cudaStream_t stream)
	{
		image::CpuVolume<DstSample> cpu_out(gpu_in.m_dim);

		const auto cpu_view_out = image::CpuVolumeView<DstSample>{
			cpu_out.Data(),
			cpu_out.Dim(),
			cpu_out.Pitch(),
			cpu_out.SlicePitch()
		};

		DownloadCompatible<DstSample>(gpu_in, cpu_view_out, stream);
		return cpu_out;
	}

	template <class T>
	inline image::CpuVolume<T> Download(const image::GpuVolumeView<T>& vol, cudaStream_t stream) {
		return DownloadCompatible<T>(vol, stream);
	}

	template <class T>
	inline image::CpuVolume<T> Download(const GpuVolume<T>& vol, cudaStream_t stream) {
		return DownloadCompatible<T>(cuda::gpu_image::MakeImageView(vol), stream);
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
		GpuImage<DstSample>& out_gpu,
		cudaStream_t stream)
	{
		UploadCompatible<DstSample>(in_cpu, cuda::gpu_image::MakeImageView(out_gpu), stream);
	}

	template <class T>
	void Upload(const image::CpuImageView<const T>& in_cpu, image::GpuImageView<T> out_gpu, cudaStream_t stream) {
		UploadCompatible<T>(in_cpu, out_gpu, stream);
	}

	template <class T>
	GpuImage<T> Create(const image::CpuImageView<T>& in_cpu, cudaStream_t stream) {
		GpuImage<T> out(in_cpu.m_dim);
		Upload(in_cpu, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}

	template <class T>
	GpuImage<T> Create(const image::CpuImage<T>& in_cpu, cudaStream_t stream) {
		GpuImage<T> out(in_cpu.m_dim);

		auto in_view = image::CpuImageView<const T>{ in_cpu.Data(), in_cpu.Dim(), size_t(in_cpu.Dim().x * sizeof(T)) };
		Upload(in_view, cuda::gpu_image::MakeImageView(out), stream);

		return out;
	}

	// 3d upload
	template <class DstSample, class SrcSample, template<class> class CpuView>
	void UploadCompatible(
		const CpuView<SrcSample>& in_cpu,
		image::GpuVolumeView<DstSample> out_gpu,
		cudaStream_t stream)
	{
		static_assert(IsUploadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for upload");

		if (in_cpu.m_dim.x != out_gpu.m_dim.x || in_cpu.m_dim.y != out_gpu.m_dim.y || in_cpu.m_dim.z != out_gpu.m_dim.z) {
			throw std::logic_error("UploadCompatible: trying to upload a volume with wrong dim");
		}

		const size_t rowBytes = in_cpu.m_dim.x * sizeof(SrcSample);
		if (in_cpu.m_pitch < rowBytes || out_gpu.m_pitch < rowBytes) {
			throw std::logic_error("UploadCompatible: pitch is smaller than row size");
		}

		const size_t sliceBytes = rowBytes * in_cpu.m_dim.y;
		if (in_cpu.m_slicePitch < sliceBytes || out_gpu.m_slicePitch < sliceBytes) {
			throw std::logic_error("UploadCompatible: slicePitch is smaller than rowBytes * dim.y");
		}

		cudaMemcpy3DParms params{};

		params.srcPtr = make_cudaPitchedPtr((void*)in_cpu.m_ptr, in_cpu.m_pitch, in_cpu.m_dim.x * sizeof(SrcSample), in_cpu.m_dim.y);
		params.dstPtr = make_cudaPitchedPtr(out_gpu.m_ptr, out_gpu.m_pitch, out_gpu.m_dim.x * sizeof(DstSample), out_gpu.m_dim.y);

		params.extent = make_cudaExtent(in_cpu.m_dim.x * sizeof(SrcSample), in_cpu.m_dim.y, in_cpu.m_dim.z);
		params.kind = cudaMemcpyHostToDevice;

		cudaCheck(cudaMemcpy3DAsync(&params, stream));
	}

	template <class DstSample, class SrcSample>
	void UploadCompatible(
		const image::CpuVolumeView<SrcSample>& in_cpu,
		GpuVolume<DstSample>& out_gpu,
		cudaStream_t stream)
	{
		UploadCompatible<DstSample>(in_cpu, cuda::gpu_image::MakeVolumeView(out_gpu), stream);
	}

	template <class T>
	void Upload(const image::CpuVolumeView<const T>& in_cpu, image::GpuVolumeView<T> out_gpu, cudaStream_t stream) {
		UploadCompatible<T>(in_cpu, out_gpu, stream);
	}

	template <class T>
	GpuVolume<T> Create(const image::CpuVolumeView<T>& in_cpu, cudaStream_t stream) {
		GpuVolume<T> out(in_cpu.m_dim);
		Upload(in_cpu, cuda::gpu_image::MakeVolumeView(out), stream);

		return out;
	}

	template <class T>
	GpuVolume<T> Create(const image::CpuVolume<T>& in_cpu, cudaStream_t stream) {
		GpuVolume<T> out(in_cpu.m_dim);

		auto in_view = image::CpuVolumeView<const T>{
			in_cpu.Data(),
			in_cpu.Dim(),
			size_t(in_cpu.Dim().x * sizeof(T)),
			size_t(in_cpu.Dim().x * sizeof(T) * in_cpu.Dim().y)
		};

		Upload(in_view, cuda::gpu_image::MakeVolumeView(out), stream);
		return out;
	}
}
