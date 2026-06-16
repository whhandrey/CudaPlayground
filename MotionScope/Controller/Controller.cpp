#include "Controller.h"
#include <queue>
#include <thread>
#include <condition_variable>

namespace loader {
	using Task = std::function<void()>;

	class ThreadPool {
	public:
		ThreadPool(int numThreads = 2)
		{
			for (int i = 0; i < numThreads; ++i) {
				m_threads.emplace_back(&ThreadPool::Run, this);
			}
		}

		~ThreadPool()
		{
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

		void AddTask(Task task) {
			{
				std::lock_guard<std::mutex> lock(m_mutex);

				if (m_stop)
					return;

				m_tasks.push(std::move(task));
			}

			m_queueCond.notify_one();
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

Controller::Controller(const std::string& folderPath)
	: m_loader(folderPath)
	, m_pool{ std::make_unique<loader::ThreadPool>() }
{
}

Controller::~Controller()
{
}
