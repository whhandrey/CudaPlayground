#include "AsyncGpuWorker.h"

namespace gpu {
	namespace worker {
		AsyncGpuWorker::AsyncGpuWorker(std::unique_ptr<IMotionGpuProcessor> processor, ProcessCallback callback)
			: m_processor{ std::move(processor) }
			, m_callback{ std::move(callback) }
			, m_thread{ std::thread(&AsyncGpuWorker::Run, this) }
		{
		}

		AsyncGpuWorker::~AsyncGpuWorker() {
			Stop();
		}

		void AsyncGpuWorker::AddJob(const Job& job) {
			{
				std::lock_guard<std::mutex> lock(m_mtx);
				if (m_stop)
					return;

				m_jobs.push(job);
			}

			m_cv.notify_one();
		}

		void AsyncGpuWorker::Run() {
			while (true) {
				Job job;

				{
					std::unique_lock<std::mutex> lock(m_mtx);
					m_cv.wait(lock, [this]() { return m_stop || !m_jobs.empty(); });

					if (m_stop && m_jobs.empty())
						return;

					job = std::move(m_jobs.front());
					m_jobs.pop();
				}

				try {
					auto confImage = m_processor->Process(job.m_prev, job.m_curr);

					if (m_callback) {
						m_callback({ job.frameIndex, job.generation, job.m_prev, job.m_curr, std::move(confImage) });
					}
				}
				catch (const std::exception&) {
					// TODO: later
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

			// TODO: clear queue?
		}
	}
}