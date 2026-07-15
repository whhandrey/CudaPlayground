#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include "../GpuProcessor/MotionViewProcessor.h"

namespace gpu::motion {
	class IGpuCommand {
	public:
		virtual ~IGpuCommand() = default;

		virtual void Execute(MotionViewProcessor& proc) = 0;
	};
}
