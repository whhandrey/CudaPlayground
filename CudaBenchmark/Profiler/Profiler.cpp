#include "Profiler.h"

void cuda::BenchProfiler::Profile(const std::string& kernelName, float ms) {
	m_runs[kernelName].push_back(ms);
}

const std::map<std::string, std::vector<float>>& cuda::BenchProfiler::GetRuns() const {
	return m_runs;
}

void cuda::BenchProfiler::Remove(const std::string& name) {
	m_runs.erase(name);
}

void cuda::BenchProfiler::Clear() {
	m_runs.clear();
}
