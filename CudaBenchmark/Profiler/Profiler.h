#pragma once
#include <Cuda/Profiler.h>
#include <map>
#include <vector>

namespace cuda {
	class BenchProfiler : public profiler::IProfiler {
	public:
		void Profile(const std::string& kernelName, float ms) override;

		const std::map<std::string, std::vector<float>>& GetRuns() const;

		void Remove(const std::string& name);
		void Clear();

	private:
		std::map<std::string, std::vector<float>> m_runs;
	};
}
