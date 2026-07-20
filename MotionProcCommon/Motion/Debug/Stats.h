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
	};

	using StatsPacket = std::map<std::string, std::vector<StatsField>>;
}
