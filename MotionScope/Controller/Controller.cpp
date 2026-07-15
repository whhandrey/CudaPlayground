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

namespace {
	template <class T>
	T wrap(T val, T minVal, T maxVal) {
		if (val > maxVal) {
			return minVal;
		}
		else if (val < minVal) {
			return maxVal;
		}

		return val;
	}
}

namespace {
	using cuda::motion::render::ViewType;

	GpuApp::ViewOption ViewTypeToViewOption(cuda::motion::render::ViewType type) {
		switch (type)
		{
		case cuda::motion::render::ViewType::ConfMap:
			return { "conf", "Confidence View" };
		case cuda::motion::render::ViewType::MotionMap:
			return { "motion_map", "Motion Map" };
		case cuda::motion::render::ViewType::MagnitudeMap:
			return { "magnitude_map", "Magnitude Map" };
		}

		throw std::logic_error("Controller: Invalid ViewType passed as parameter");
	}

	ViewType ViewTypeFromId(const std::string& id) {
		if (id == "conf") {
			return ViewType::ConfMap;
		}

		else if (id == "motion_map") {
			return ViewType::MotionMap;
		}

		else if (id == "magnitude_map") {
			return ViewType::MagnitudeMap;
		}

		throw std::logic_error("Controller: Invalid id passed as parameter");
	}
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

	void Controller::SetPlayMode(PlayMode mode)
	{
		m_playMode = mode;
	}

	bool Controller::TryStepForward()
	{
		const auto range = GetFramesRange();

		// if we are already at the max frame (normal mode) and trying to do step forward -> no move
		if (m_playMode == PlayMode::Normal && m_currentIndex == range.second) {
			return false;
		}

		int nextIndex = m_currentIndex + 1;
		if (m_playMode == PlayMode::Loop) {
			nextIndex = wrap(nextIndex, range.first, range.second);
		}
		
		SetFrame(nextIndex);
		return true;
	}

	bool Controller::TryStepBackward()
	{
		const auto range = GetFramesRange();

		// if we are already at the min frame (normal mode) and trying to do step back -> no move
		if (m_playMode == PlayMode::Normal && m_currentIndex == range.first) {
			return false;
		}

		int nextIndex = m_currentIndex - 1;
		if (m_playMode == PlayMode::Loop) {
			nextIndex = wrap(nextIndex, range.first, range.second);
		}

		SetFrame(nextIndex);
		return true;
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

	void Controller::SetFrame(int index)
	{
		const auto range = GetFramesRange();
		m_currentIndex = std::clamp(index, range.first, range.second);

		RequestFrame(m_currentIndex);
		RequestFrame(m_currentIndex + 1);
	}

	void Controller::SetView(ViewSlot slot, const std::string& id)
	{
		switch (slot)
		{
		case GpuApp::ViewSlot::View1:
			m_views.view1 = ViewTypeFromId(id);
			return;
		case GpuApp::ViewSlot::View2:
			m_views.view2 = ViewTypeFromId(id);
			return;
		}

		throw std::logic_error("Controller::SetView: invalid view slot");
	}

	std::pair<int, int> Controller::GetFramesRange() const
	{
		return std::make_pair(0, int(m_loader->NumImages()) - 2);
	}

	int Controller::GetCurrentIndex() const
	{
		return m_currentIndex;
	}

	std::vector<ViewOption> Controller::AllViewOptions() const
	{
		const auto allViews = cuda::motion::render::AllViews();

		std::vector<ViewOption> output;
		std::transform(allViews.begin(), allViews.end(), std::back_inserter(output), [](ViewType type) {
			return ViewTypeToViewOption(type);
		});

		return output;
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
			m_cache[m_currentIndex + 1],
			{ m_views.view1, m_views.view2 }
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
			result.views[m_views.view1],
			result.views[m_views.view2]
		);
	}
}
