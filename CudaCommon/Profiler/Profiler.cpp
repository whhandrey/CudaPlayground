#include "Profiler.h"

namespace cuda::profile {
	void SampledProfiler::Profile(const std::string& kernelName, float ms) {
		m_runs[kernelName].push_back(ms);
	}

	const std::map<std::string, std::vector<float>>& SampledProfiler::GetRuns() const {
		return m_runs;
	}

	void SampledProfiler::Remove(const std::string& name) {
		m_runs.erase(name);
	}

	void SampledProfiler::Clear() {
		m_runs.clear();
	}

	void SimpleProfiler::Profile(const std::string& kernelName, float ms) {
		m_runs[kernelName] = ms;
	}

	const std::map<std::string, float>& SimpleProfiler::GetRuns() const {
		return m_runs;
	}

	void SimpleProfiler::Clear() {
		m_runs.clear();
	}
}
