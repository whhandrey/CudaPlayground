#pragma once
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>
#include <map>
#include <QImage>
#include "../GpuProcessor/MotionViewProcessor.h"

namespace gpu {
	namespace motion {
		using cuda::motion::render::ViewType;

		struct Job {
			int frameIndex;
			size_t generation;
			QImage prev;
			QImage curr;
			std::vector<ViewType> requestedViews;
		};

		struct Result {
			int frameIndex;
			size_t generation;
			QImage prev;
			QImage curr;
			std::map<ViewType, QImage> views;
		};

		using ProcessCallback = std::function<void(Result&&)>;

		class AsyncGpuWorker {
		public:
			AsyncGpuWorker(std::unique_ptr<MotionViewProcessor> processor);
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
			std::unique_ptr<MotionViewProcessor> m_processor;
			ProcessCallback m_callback = nullptr;

			bool m_stop = false;

			std::thread m_thread;
			std::mutex m_mtx;
			std::condition_variable m_cv;
			std::queue<Job> m_jobs;
		};
	}
}
