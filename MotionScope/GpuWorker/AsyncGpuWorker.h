#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <QImage>
#include <GpuProcessor/IMotionGpuProcessor.h>

namespace gpu {
	namespace motion {
		struct Job {
			size_t frameIndex;
			size_t generation;
			QImage m_prev;
			QImage m_curr;
		};

		struct Result {
			size_t frameIndex;
			size_t generation;
			QImage m_prev;
			QImage m_curr;
			QImage m_conf;
		};

		using ProcessCallback = std::function<void(Result&&)>;

		class AsyncGpuWorker {
		public:
			AsyncGpuWorker(std::unique_ptr<cuda::IMotionGpuProcessor> processor);
			~AsyncGpuWorker();

			AsyncGpuWorker(const AsyncGpuWorker&) = delete;
			AsyncGpuWorker& operator=(const AsyncGpuWorker&) = delete;

		public:
			void AddJob(const Job& job);
			void SetCallback(ProcessCallback&& callback);

		private:
			void Run();
			void Stop();

		private:
			std::unique_ptr<cuda::IMotionGpuProcessor> m_processor;
			ProcessCallback m_callback = nullptr;

			bool m_stop = false;

			std::thread m_thread;
			std::mutex m_mtx;
			std::condition_variable m_cv;
			std::queue<Job> m_jobs;
		};
	}
}
