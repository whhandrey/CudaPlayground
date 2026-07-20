#pragma once
#include <Cuda/Motion/BlockMatchingParams.h>
#include <Motion/Debug/Stats.h>
#include <Motion/Debug/Callback.h>
#include <map>

namespace cuda::motion::debug {
	using namespace ::motion::debug;

	class Collector {
	public:
		Collector(StatsCallback&& callback);

		void AddParams(const BlockMatchingParams& p);
		void AddStat(const std::string& scope, const std::string& group, const std::string& name, float time);

		void IssueCallback();

	private:
		StatsCallback m_callback;
		StatsPacket m_stats;
	};
}
