#pragma once
#include "../Common/Common.h"
#include <vector>
#include <string>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

namespace Image {
	namespace traits {
		template <class T>
		struct PixelTraits;

		template <>
		struct PixelTraits<unsigned char> {
			static constexpr size_t Channels = 1;
		};

		template <>
		struct PixelTraits<uchar4> {
			static constexpr size_t Channels = 4;
		};
	}

	namespace read {
		template <class T>
		T ReadPixel(const unsigned char* p);

		template <>
		inline unsigned char ReadPixel<unsigned char>(const unsigned char* p) {
			return p[0];
		}

		template <>
		inline uchar4 ReadPixel<uchar4>(const unsigned char* p) {
			return uchar4{ p[0], p[1], p[2], p[3] };
		}
	}
}

template <class T>
class ImageCPU {
public:
	ImageCPU(ivec2 dim)
		: m_dim(dim)
		, m_data(dim.x * dim.y)
	{
	}

	ImageCPU(unsigned char* ptr, ivec2 dim)
		: ImageCPU(dim)
	{
		constexpr size_t channels = Image::traits::PixelTraits<T>::Channels;

		for (size_t i = 0; i < dim.x * dim.y; ++i) {
			const unsigned char* p = ptr + i * channels;
			m_data[i] = Image::read::ReadPixel<T>(p);
		}
	}

	ImageCPU(const ImageCPU&) = delete;
	ImageCPU& operator=(const ImageCPU&) = delete;

	ImageCPU(ImageCPU&&) noexcept = default;
	ImageCPU& operator=(ImageCPU&&) noexcept = default;

	ivec2 Dim() const {
		return m_dim;
	}

	const T* Data() const {
		return m_data.data();
	}

	T* Data() {
		return m_data.data();
	}

private:
	ivec2 m_dim;
	std::vector<T> m_data;
};

template <class T>
class ImageGPU {
public:
	ImageGPU(ivec2 dim)
		: m_dim(dim)
	{
		Create();
	}

	ImageGPU(const T* ptr, ivec2 dim, cudaStream_t stream)
		: ImageGPU(dim)
	{
		cudaCheck(cudaMemcpy2DAsync(m_ptr, m_pitch, ptr, dim.x * sizeof(T), dim.x * sizeof(T), dim.y, cudaMemcpyHostToDevice, stream));
	}

	ImageGPU(const ImageCPU<T>& img, cudaStream_t stream)
		: ImageGPU<T>(img.Data(), img.Dim(), stream)
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

	ivec2 Dim() const {
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
	ivec2 m_dim;
	size_t m_pitch = 0;
	T* m_ptr = nullptr;
};

namespace Image {
	template <class T>
	inline ImageCPU<T> LoadFromFile(const std::string& file, int desiredChannels = 4) {
		int width, height, channels;
		unsigned char* img = stbi_load(file.c_str(), &width, &height, &channels, 4);

		const ivec2 dim{ static_cast<unsigned int>(width), static_cast<unsigned int>(height) };
		ImageCPU<T> output(img, dim);

		stbi_image_free(img);
		return output;
	}

	template <class T>
	inline void WriteToFile(const ImageCPU<T>& img, const std::string& file) {
		const int numChannels = 4;
		stbi_write_png(file.c_str(), img.Dim().x, img.Dim().y, numChannels, img.Data(), img.Dim().x * sizeof(T));
	}

	template <class T>
	inline ImageCPU<T> ImageGpuToCpu(const ImageGPU<T>& img, cudaStream_t stream) {
		ImageCPU<T> output(img.Dim());

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
