#pragma once
#include <string>

namespace cuda {
	namespace profile {
		class IProfiler {
		public:
			virtual ~IProfiler() = default;
			virtual void Profile(const std::string& kernelName, float ms) = 0;
		};
	}
}
