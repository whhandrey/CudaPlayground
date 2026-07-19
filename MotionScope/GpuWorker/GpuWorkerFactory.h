#pragma once
#include "AsyncGpuWorker.h"
#include <Debug/Callback.h>

namespace app::worker {
	using ::motion::debug::StatsCallback;

	class GpuWorkerFactory {
	public:
		std::unique_ptr<AsyncGpuWorker> Create(StatsCallback&& callback);
	};
}
