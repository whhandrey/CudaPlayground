#pragma once
#include "IProfiler.h"
#include <map>
#include <vector>

namespace cuda::profile {
	class SampledProfiler : public IProfiler {
	public:
		void Profile(const std::string& kernelName, float ms) override;

		const std::map<std::string, std::vector<float>>& GetRuns() const;

		void Remove(const std::string& name);
		void Clear();

	private:
		std::map<std::string, std::vector<float>> m_runs;
	};

	class SimpleProfiler : public IProfiler {
	public:
		void Profile(const std::string& kernelName, float ms) override;

		const std::map<std::string, float>& GetRuns() const;

		void Clear();

	private:
		std::map<std::string, float> m_runs;
	};
}
