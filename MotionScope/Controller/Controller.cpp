#include "Controller.h"
#include "../ImageLoader/ImageLoader.h"
#include "../GpuWorker/AsyncGpuWorker.h"
#include <queue>
#include <thread>
#include <condition_variable>

namespace loader {
	using Task = std::function<void()>;

	const int defThreadsNum = 2;

	class ThreadPool {
	public:
		ThreadPool(int numThreads = defThreadsNum)
		{
			for (int i = 0; i < numThreads; ++i) {
				m_threads.emplace_back(&ThreadPool::Run, this);
			}
		}

		~ThreadPool()
		{
			Stop();
		}

		void AddTask(Task task) {
			{
				std::lock_guard<std::mutex> lock(m_mutex);

				if (m_stop)
					return;

				m_tasks.push(std::move(task));
			}

			m_queueCond.notify_one();
		}

		void Stop() {
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_stop = true;
			}

			m_queueCond.notify_all();

			for (auto& th : m_threads) {
				if (th.joinable()) {
					th.join();
				}
			}
		}

	private:
		void Run() {
			while (true) {
				Task task;

				{
					std::unique_lock<std::mutex> lock(m_mutex);
					m_queueCond.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });

					if (m_stop && m_tasks.empty()) {
						return;
					}

					task = std::move(m_tasks.front());
					m_tasks.pop();
				}

				task();
			}
		}

	private:
		std::vector<std::thread> m_threads;
		bool m_stop = false;

		std::mutex m_mutex;

		std::queue<Task> m_tasks;
		std::condition_variable m_queueCond;
	};
}

namespace GpuApp {
	Controller::Controller(std::unique_ptr<gpu::motion::AsyncGpuWorker> gpuWorker)
		: m_pool{ std::make_unique<loader::ThreadPool>() }
		, m_gpuWorker{ std::move(gpuWorker) }
	{
		m_gpuWorker->SetCallback([this](gpu::motion::Result&& result) mutable {
			QMetaObject::invokeMethod(
				this, [this, res = std::move(result)]() mutable {
					OnGpuResultReady(std::move(res));
				},
				Qt::QueuedConnection
			);
		});
	}

	Controller::~Controller()
	{
	}

	void Controller::SetFolder(const std::string& folderPath)
	{
		++m_generation;

		m_pool->Stop();

		m_loader = std::make_unique<image::Loader>(folderPath);
		m_pool = std::make_unique<loader::ThreadPool>();

		m_cache = std::vector<QImage>(m_loader->NumImages());
		m_currentIndex = 0;

		RequestFrame(0);
		RequestFrame(1);
	}

	void Controller::RequestFrame(int index)
	{
		const size_t generation = m_generation;

		m_pool->AddTask([this, index, generation]() {
			auto img = m_loader->Load(index);

			QMetaObject::invokeMethod(this, [this, index, generation, frame = std::move(img)]() mutable {
				if (generation != m_generation)
					return;

				OnFrameReady(index, generation, std::move(frame));
			},
			Qt::QueuedConnection);
		});
	}

	void Controller::OnFrameReady(int index, size_t generation, QImage&& image)
	{
		if (generation != m_generation)
			return;

		m_cache[index] = std::move(image);
		TryProcessImagePair();
	}

	void Controller::TryProcessImagePair()
	{
		if (m_currentIndex + 1 >= m_loader->NumImages())
			return;

		if (m_cache[m_currentIndex].isNull() || m_cache[m_currentIndex + 1].isNull())
			return;

		auto job = gpu::motion::Job {
			m_currentIndex,
			m_generation,
			m_cache[m_currentIndex],
			m_cache[m_currentIndex + 1]
		};

		m_gpuWorker->AddJob(job);
	}

	void Controller::OnGpuResultReady(gpu::motion::Result result)
	{
		using cuda::motion::render::ViewType;

		if (result.generation != m_generation)
			return;

		if (result.frameIndex != m_currentIndex)
			return;

		emit ImagesReady(
			result.prev,
			result.curr,
			result.views[ViewType::ConfMap],
			result.views[ViewType::MotionMap]
		);
	}
}
