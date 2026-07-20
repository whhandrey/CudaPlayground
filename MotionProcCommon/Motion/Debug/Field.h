#pragma once
#include <string>
#include <variant>
#include <Image/ImageTypes.h>

namespace motion::debug {
	using Value = std::variant<
		int,
		unsigned int,
		float,
		image::vec2i,
		image::vec2ui,
		std::string
	>;

	struct Field {
		std::string name;
		std::string group;

		Value value;
		std::string unit;
	};
}
