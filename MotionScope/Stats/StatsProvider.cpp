#include "StatsProvider.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace detail {
	using ::motion::debug::StatsValue;
	using ::motion::debug::Unit;

	std::string ToString(const StatsValue& value) {
		auto getVariant = [](const auto& v) -> std::string {
			using T = std::decay_t<decltype(v)>;

			if constexpr (std::is_same_v<T, int>) {
				return std::to_string(v);
			}
			else if constexpr (std::is_same_v<T, unsigned int>) {
				return std::to_string(v);
			}
			else if constexpr (std::is_same_v<T, float>) {
				return std::to_string(v);
			}
			else if constexpr (std::is_same_v<T, std::string>) {
				return v;
			}
			else if constexpr (std::is_same_v<T, image::vec2i>) {
				return "(" + std::to_string(v.x) + "; " + std::to_string(v.y) + ")";
			}
			else if constexpr (std::is_same_v<T, image::vec2ui>) {
				return "(" + std::to_string(v.x) + "; " + std::to_string(v.y) + ")";
			}
			else {
				static_assert(std::is_same_v<T, void>, "Unsupported StatsValue type");
			}
		};

		return std::visit(getVariant, value);
	}

	std::string UnitToString(Unit unit) {
		switch (unit)
		{
		case motion::debug::Unit::px:
			return "px";
		case motion::debug::Unit::ms:
			return "ms";
		case motion::debug::Unit::NoUnit:
			return std::string();
		}

		throw std::logic_error("StatsProvider::UnitToString: unsupported unit");
	}
}

namespace app {
	StatsProvider::StatsProvider(const StatsPacket& stats)
		: m_stats{ stats }
	{
	}

	std::map<std::string, std::vector<std::string>> StatsProvider::GetFieldsByGroups() const {
		std::map<std::string, std::vector<std::string>> output;

		for (const auto& statPair : m_stats) {
			std::vector<std::string> statNames;

			std::transform(statPair.second.begin(), statPair.second.end(), std::back_inserter(statNames), [](const auto& stat) {
				return stat.name;
			});

			output[statPair.first] = std::move(statNames);
		}

		return output;
	}

	ValueUnit StatsProvider::GetStatValue(const std::string& group, const std::string& name) {
		const auto& bucket = m_stats.at(group);

		auto it = std::find_if(bucket.begin(), bucket.end(), [&name](const auto& statValue) {
			return statValue.name == name;
		});

		if (it == bucket.end()) {
			throw std::logic_error("StatsProvider::ToString: invalid statValue name");
		}

		return { detail::ToString(it->value) , detail::UnitToString(it->unit) };
	}
}
