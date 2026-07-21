#pragma once
#include <string>
#include <Motion/Debug/Stats.h>

namespace app {
	using ::motion::debug::StatsPacket;

	using ValueUnit = std::pair<std::string, std::string>;

	class StatsProvider {
	public:
		explicit StatsProvider(const StatsPacket& stats);

		std::map<std::string, std::vector<std::string>> GetFieldsByGroups() const;
		ValueUnit GetStatValue(const std::string& group, const std::string& name);

	private:
		StatsPacket m_stats;
	};
}
