#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include "IMotionGpuProcessor.h"

namespace gpu {
	namespace worker {
		struct Job {
			int frameIndex;
			size_t generation;
			ImageViewRGBA8 m_prev;
			ImageViewRGBA8 m_curr;
		};

		struct Result {
			int frameIndex;
			size_t generation;
			ImageViewRGBA8 m_prev;
			ImageViewRGBA8 m_curr;
			ImageRGBA8 m_conf;
		};

		using ProcessCallback = std::function<void(Result)>;

		class AsyncGpuWorker {
		public:
			AsyncGpuWorker(std::unique_ptr<IMotionGpuProcessor> processor, ProcessCallback callback);
			~AsyncGpuWorker();

			AsyncGpuWorker(const AsyncGpuWorker&) = delete;
			AsyncGpuWorker& operator=(const AsyncGpuWorker&) = delete;

		public:
			void AddJob(const Job& job);

		private:
			void Run();
			void Stop();

		private:
			std::unique_ptr<IMotionGpuProcessor> m_processor;
			ProcessCallback m_callback;

			bool m_stop = false;

			std::thread m_thread;
			std::mutex m_mtx;
			std::condition_variable m_cv;
			std::queue<Job> m_jobs;
		};
	}
}
