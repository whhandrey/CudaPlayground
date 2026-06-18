#pragma once
#include "Logger.h"

namespace cuda {
	void KernelLogger::Log(const std::string& name, float ms) {
		m_runs[name].push_back(ms);
	}

	const std::map<std::string, std::vector<float>>& KernelLogger::GetRuns() const {
		return m_runs;
	}

	void KernelLogger::Remove(const std::string& name) {
		m_runs.erase(name);
	}

	void KernelLogger::Clear() {
		m_runs.clear();
	}
}
