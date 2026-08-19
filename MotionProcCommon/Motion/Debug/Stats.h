#pragma once
#include <string>
#include <variant>
#include <map>
#include <vector>
#include <Image/ImageTypes.h>

namespace motion::debug {
	using StatsValue = std::variant<
		int,
		unsigned int,
		float,
		image::vec2i,
		image::vec2ui,
		std::string
	>;

	enum class Unit {
		px,
		ms,
		NoUnit
	};

	struct StatsField {
		std::string name;
		StatsValue value;
		Unit unit;

		bool operator==(const StatsField& rhs) const {
			return name == rhs.name
				&& value == rhs.value
				&& unit == rhs.unit;
		}
	};

	using StatsPacket = std::map<std::string, std::vector<StatsField>>;
}
