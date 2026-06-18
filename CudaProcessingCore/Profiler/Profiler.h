#pragma once
#include <string>

namespace cuda {
	namespace profiler {
		class IProfiler {
		public:
			virtual ~ILogger() = default;
			virtual void Log(const std::string& name, float ms) = 0;
		};
	}
}

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


