#include "AsyncGpuWorker.h"
#include "GpuWorkerFactory.h"
#include "../GpuProcessor/MotionViewProcessor.h"

namespace app::worker {
	std::unique_ptr<AsyncGpuWorker> app::worker::GpuWorkerFactory::Create(StatsCallback&& callback) {
		using app::motion::MotionViewProcessor;

		auto rawProc = cuda::motion::IMotionViewProcessor::Create(std::move(callback));
		auto motionProc = std::make_unique<MotionViewProcessor>(std::move(rawProc));

		return std::make_unique<AsyncGpuWorker>(std::move(motionProc));
	}
}
