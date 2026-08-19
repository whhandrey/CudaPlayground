#pragma once
#include <type_traits>

#define ALIGN(num) alignas(num)

#define CHECK_SIZE(T, N) \
    static_assert(sizeof(T) == (N), #T " has wrong size")

#define CHECK_ALIGN(T, N) \
    static_assert(alignof(T) == (N), #T " has wrong alignment")

#define CHECK_TRIVIAL(T) \
    static_assert(std::is_trivially_copyable<T>::value, #T " must be trivially copyable")

#define CHECK_STANDARD_LAYOUT(T) \
    static_assert(std::is_standard_layout<T>::value, #T " must be standard layout")

namespace image {
	using uchar1 = unsigned char;

	struct ALIGN(4) vec4uc {
		unsigned char x;
		unsigned char y;
		unsigned char z;
		unsigned char w;
	};

	struct ALIGN(8) vec2ui {
		unsigned int x;
		unsigned int y;

		bool operator==(const vec2ui& rhs) const {
			return x == rhs.x && y == rhs.y;
		}
	};

	struct ALIGN(8) vec2i {
		int x;
		int y;

		bool operator==(const vec2i& rhs) const {
			return x == rhs.x && y == rhs.y;
		}
	};

	struct ALIGN(16) vec4ui {
		unsigned int x;
		unsigned int y;
		unsigned int z;
		unsigned int w;
	};

	struct ALIGN(16) vec4i {
		int x;
		int y;
		int z;
		int w;
	};

	struct ALIGN(16) vec4f {
		float x;
		float y;
		float z;
		float w;
	};

	CHECK_SIZE(uchar1, 1);
	CHECK_ALIGN(uchar1, 1);

	CHECK_SIZE(vec4uc, 4);
	CHECK_ALIGN(vec4uc, 4);

	CHECK_SIZE(vec2ui, 8);
	CHECK_ALIGN(vec2ui, 8);

	CHECK_SIZE(vec2i, 8);
	CHECK_ALIGN(vec2i, 8);

	CHECK_SIZE(vec4ui, 16);
	CHECK_ALIGN(vec4ui, 16);

	CHECK_SIZE(vec4i, 16);
	CHECK_ALIGN(vec4i, 16);

	CHECK_SIZE(vec4f, 16);
	CHECK_ALIGN(vec4f, 16);

	CHECK_TRIVIAL(uchar1);
	CHECK_TRIVIAL(vec4uc);
	CHECK_TRIVIAL(vec2ui);
	CHECK_TRIVIAL(vec2i);
	CHECK_TRIVIAL(vec4ui);
	CHECK_TRIVIAL(vec4i);
	CHECK_TRIVIAL(vec4f);

	CHECK_STANDARD_LAYOUT(uchar1);
	CHECK_STANDARD_LAYOUT(vec4uc);
	CHECK_STANDARD_LAYOUT(vec2ui);
	CHECK_STANDARD_LAYOUT(vec2i);
	CHECK_STANDARD_LAYOUT(vec4ui);
	CHECK_STANDARD_LAYOUT(vec4i);
	CHECK_STANDARD_LAYOUT(vec4f);
}
