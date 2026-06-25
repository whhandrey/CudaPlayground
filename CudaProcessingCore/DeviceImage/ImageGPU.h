#pragma once
#include <Image/Image.h>
#include <Image/ImageView.h>
#include <Cuda/CudaCheck.h>

template <class T>
class ImageGPU {
public:
	ImageGPU() = default;

	ImageGPU(image::vec2ui dim)
		: m_dim(dim)
	{
		Create();
	}

	ImageGPU(const image::ImageView<T>& img, cudaStream_t stream)
		: ImageGPU(img.m_dim)
	{
		Upload(img, stream);
	}

	ImageGPU(const T* ptr, image::vec2ui dim, cudaStream_t stream)
		: ImageGPU<T>(image::ImageView<T>{ ptr, dim, dim.x * sizeof(T) }, stream)
	{
	}

	ImageGPU(const image::Image<T>& img, cudaStream_t stream)
		: ImageGPU<T>(image::ImageView<T>{ img.Data(), img.Dim(), img.Dim().x * sizeof(T) }, stream)
	{
	}

	ImageGPU(const ImageGPU&) = delete;
	ImageGPU& operator=(const ImageGPU&) = delete;

	ImageGPU(ImageGPU&& other) noexcept
		: m_dim(other.m_dim)
		, m_pitch(other.m_pitch)
		, m_ptr(other.m_ptr)
	{
		other.m_dim = {};
		other.m_pitch = 0;
		other.m_ptr = nullptr;
	}

	ImageGPU& operator=(ImageGPU&& other) noexcept {
		if (this != &other) {
			Destroy();

			m_dim = other.m_dim;
			m_pitch = other.m_pitch;
			m_ptr = other.m_ptr;

			other.m_dim = {};
			other.m_pitch = 0;
			other.m_ptr = nullptr;
		}

		return *this;
	}

	image::vec2ui Dim() const {
		return m_dim;
	}

	size_t Pitch() const {
		return m_pitch;
	}

	const T* Data() const {
		return m_ptr;
	}

	T* Data() {
		return m_ptr;
	}

	template <class SrcSample>
	void UploadCompatible(const image::ImageView<SrcSample>& img, cudaStream_t stream) {
		static_assert(sizeof(T) == sizeof(SrcSample), "Source and destination pixel sizes must match for raw upload");

		if (m_dim.x != img.m_dim.x || m_dim.y != img.m_dim.y) {
			throw std::logic_error("ImageGPU::Upload: trying to upload an image with wrong dim!");
		}

		cudaCheck(cudaMemcpy2DAsync(m_ptr, m_pitch, img.m_ptr, img.m_pitch, img.m_dim.x * sizeof(SrcSample), img.m_dim.y, cudaMemcpyHostToDevice, stream));
	}

	void Upload(const image::ImageView<T>& img, cudaStream_t stream) {
		UploadCompatible<T>(img, stream);
	}

	~ImageGPU() {
		Destroy();
	}

private:
	void Create() {
		cudaCheck(cudaMallocPitch(&m_ptr, &m_pitch, m_dim.x * sizeof(T), m_dim.y));
	}

	void Destroy() {
		if (m_ptr) {
			cudaFree(m_ptr);
			m_ptr = nullptr;
		}

		m_pitch = 0;
		m_dim = {};
	}

private:
	image::vec2ui m_dim = {};
	size_t m_pitch = 0;
	T* m_ptr = nullptr;
};
