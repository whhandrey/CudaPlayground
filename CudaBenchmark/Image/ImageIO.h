#pragma once
#include <string>
#include <Image/Image.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <vector_types.h>

namespace image {
	namespace traits {
		template <class T>
		struct PixelTraits;

		template <>
		struct PixelTraits<unsigned char> {
			static constexpr int channels = 1;
		};

		template <>
		struct PixelTraits<uchar4> {
			static constexpr int channels = 4;
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

	template <class T>
	inline image::Image<T> LoadFromFile(const std::string& file) {
		int width, height, channels;

		int channelsToRead = traits::PixelTraits<T>::channels;
		unsigned char* img = stbi_load(file.c_str(), &width, &height, &channels, channelsToRead);

		const image::vec2ui dim{ static_cast<unsigned int>(width), static_cast<unsigned int>(height) };
		image::Image<T> output(img, dim);

		stbi_image_free(img);
		return output;
	}

	template <class T>
	inline void WriteToFile(const image::Image<T>& img, const std::string& file) {
		const int numChannels = traits::PixelTraits<T>::channels;
		stbi_write_png(file.c_str(), img.Dim().x, img.Dim().y, numChannels, img.Data(), img.Dim().x * sizeof(T));
	}
}
