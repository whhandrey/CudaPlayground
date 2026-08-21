#pragma once
#include <Cuda/Motion/BlockMatchingParams.h>
#include <Motion/Debug/Stats.h>
#include <map>

namespace cuda::motion::debug {
	using ::motion::debug::StatsPacket;

	class Collector {
	public:
		void AddParams(const BlockMatchingParams& p);
		void AddStat(const std::string& scope, const std::string& name, float time);

		StatsPacket TakeStats();

	private:
		StatsPacket m_stats;
	};
}
