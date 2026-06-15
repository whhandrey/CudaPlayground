#pragma once
#include <string>
#include <vector>
#include <map>
#include "Common.h"

namespace cuda {
	class KernelLogger {
	public:
		void Log(const std::string& name, float ms);

		const std::map<std::string, std::vector<float>>& GetRuns() const;

		void Remove(const std::string& name);
		void Clear();

	private:
		std::map<std::string, std::vector<float>> m_runs;
	};

	inline KernelLogger& Logger() {
		static KernelLogger s_logger;
		return s_logger;
	}
}

namespace util {
	inline std::string BlockDimToString(const ivec2& blockDim) {
		return "(" + std::to_string(blockDim.x) + ", " + std::to_string(blockDim.y) + ")";
	}
}
