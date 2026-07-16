#include "AsyncGpuWorker.h"
#include <Image/Image.h>

namespace gpu {
	namespace motion {
		AsyncGpuWorker::AsyncGpuWorker(std::unique_ptr<MotionViewProcessor> processor)
			: m_processor{ std::move(processor) }
			, m_thread{ std::thread(&AsyncGpuWorker::Run, this) }
		{
		}

		AsyncGpuWorker::~AsyncGpuWorker() {
			Stop();
		}

		void AsyncGpuWorker::AddJob(IGpuJob::Ptr&& job) {
			{
				std::lock_guard<std::mutex> lock(m_mtx);
				if (m_stop)
					return;

				m_jobs.push(std::move(job));
			}

			m_cv.notify_one();
		}

		void AsyncGpuWorker::Run() {
			while (true) {
				IGpuJob::Ptr job;

				{
					std::unique_lock<std::mutex> lock(m_mtx);
					m_cv.wait(lock, [this]() { return m_stop || !m_jobs.empty(); });

					if (m_stop && m_jobs.empty())
						return;

					job = std::move(m_jobs.front());
					m_jobs.pop();
				}

				try {
					job->Execute(*m_processor);
				}
				catch (const std::exception& /*ex*/) {
					// m_exceptionMgr->Report(ex);
				}
			}
		}

		void AsyncGpuWorker::Stop() {
			{
				std::lock_guard<std::mutex> lock(m_mtx);
				m_stop = true;
			}

			m_cv.notify_one();

			if (m_thread.joinable()) {
				m_thread.join();
			}

			// TODO: clear queue when move slider?
		}
	}
}