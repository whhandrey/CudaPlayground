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

		void AddCpuStat(const std::string& name, float time, Unit unit = Unit::NoUnit);
		void AddGpuStat(const std::string& name, float time, Unit unit = Unit::NoUnit);

		void IssueCallback();

	private:
		void Clear();

	private:
		StatsCallback m_callback;
		StatsPacket m_stats;
	};
}
