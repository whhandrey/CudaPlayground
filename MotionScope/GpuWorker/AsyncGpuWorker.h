#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include "../GpuProcessor/MotionViewProcessor.h"
#include "Job/GpuJob.h"

namespace app::worker {
	using cuda::motion::render::ViewType;

	class AsyncGpuWorker {
	public:
		AsyncGpuWorker(std::unique_ptr<motion::MotionViewProcessor> processor);
		~AsyncGpuWorker();

		AsyncGpuWorker(const AsyncGpuWorker&) = delete;
		AsyncGpuWorker& operator=(const AsyncGpuWorker&) = delete;

	public:
		void AddJob(motion::IGpuJob::Ptr&& job);
		void Stop();

	private:
		void Run();

	private:
		std::unique_ptr<motion::MotionViewProcessor> m_processor;
		std::queue<motion::IGpuJob::Ptr> m_jobs;

		bool m_stop = false;

		std::thread m_thread;
		std::mutex m_mtx;
		std::condition_variable m_cv;
	};
}
