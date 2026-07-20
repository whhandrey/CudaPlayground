#pragma once
#include <Cuda/Motion/BlockMatchingParams.h>
#include <Debug/StatsProvider.h>
#include <Debug/Callback.h>

namespace cuda::motion::debug {
	using ::motion::debug::StatsCallback;

	class Collector {
	public:
		Collector(StatsCallback&& callback);

		void AddParams(const BlockMatchingParams& p);

		void AddCpuStat(const std::string& name, float time);
		void AddGpuStat(const std::string& name, float time);

		void IssueCallback();

	private:
		void Clear();

	private:
		StatsCallback m_callback;

		BlockMatchingParams m_params = {};

		std::map<std::string, float> m_cpuStats;
		std::map<std::string, float> m_gpuStats;
	};
}
