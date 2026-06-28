#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include <GpuProcessor/IMotionViewProcessor.h>

namespace gpu {
	namespace motion {
		using cuda::motion::render::ViewType;

		struct Job {
			size_t frameIndex;
			size_t generation;
			QImage prev;
			QImage curr;
		};

		struct Result {
			size_t frameIndex;
			size_t generation;
			QImage prev;
			QImage curr;
			std::map<ViewType, QImage> views;
		};

		using ProcessCallback = std::function<void(Result&&)>;

		class AsyncGpuWorker {
		public:
			AsyncGpuWorker(std::unique_ptr<cuda::motion::IMotionViewProcessor> processor);
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
			std::unique_ptr<cuda::motion::IMotionViewProcessor> m_processor;
			ProcessCallback m_callback = nullptr;

			bool m_stop = false;

			std::thread m_thread;
			std::mutex m_mtx;
			std::condition_variable m_cv;
			std::queue<Job> m_jobs;
		};
	}
}
