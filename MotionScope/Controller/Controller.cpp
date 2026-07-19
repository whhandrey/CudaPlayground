#include "Controller.h"
#include "../ImageLoader/ImageLoader.h"
#include "../GpuWorker/AsyncGpuWorker.h"
#include "../GpuWorker/Job/GpuJob.h"
#include <queue>
#include <thread>
#include <condition_variable>

namespace pool {
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

	app::ViewOption ViewTypeToViewOption(cuda::motion::render::ViewType type) {
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

namespace detail {
	constexpr int minOffset = 1;
}

namespace app {
	Controller::Controller(app::worker::GpuWorkerFactory& factory)
		: m_pool{ std::make_unique<pool::ThreadPool>() }
	{
		m_gpuWorker = factory.Create([this](IStatsProvider::Ptr statsProvider) mutable {
			QMetaObject::invokeMethod(
				this, [this, provider = std::move(statsProvider)]() mutable {
					OnDebugStats(std::move(provider));
				},
				Qt::QueuedConnection
			);
		});
	}

	Controller::~Controller()
	{
		m_gpuWorker->Stop();
	}

	void Controller::SetFolder(const std::string& folderPath)
	{
		++m_generation;

		m_pool->Stop();

		m_loader = std::make_unique<image::Loader>(folderPath);
		m_pool = std::make_unique<pool::ThreadPool>();

		m_cache = std::vector<QImage>(m_loader->NumImages());

		m_currentIndex = 0;
		m_pairOffset = detail::minOffset;

		RequestPair({ m_currentIndex, m_currentIndex + m_pairOffset });
	}

	void Controller::SetPlayMode(PlayMode mode)
	{
		m_playMode = mode;
	}

	void Controller::SetPlayState(PlayState state)
	{
		m_playState = state;
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

	void Controller::PreparePlaybackStart()
	{
		const auto range = GetFramesRange();

		if (m_playMode == PlayMode::Normal && m_currentIndex == range.second) {
			SetFrame(range.first);
		}
	}

	void Controller::RequestPair(app::FramePair reqPair)
	{
		m_requestedPair = reqPair;
		m_requestPairSubmitted = false;

		RequestFrame(reqPair.first);
		RequestFrame(reqPair.second);

		TryAnalyseImagePair();
	}

	void Controller::RequestFrame(int index)
	{
		if (!m_cache[index].isNull()) {
			return;
		}

		const size_t generation = m_generation;

		m_pool->AddTask([this, index, generation]() {
			auto img = m_loader->Load(index);

			QMetaObject::invokeMethod(this, [this, index, generation, img = std::move(img)]() mutable {
				OnFrameReady(index, generation, std::move(img));
			},
			Qt::QueuedConnection);
		});
	}

	void Controller::SetFrame(int index)
	{
		const auto range = GetFramesRange();
		m_currentIndex = std::clamp(index, range.first, range.second);

		const auto offsetRange = GetOffsetRange();
		m_pairOffset = std::clamp(m_pairOffset, offsetRange.first, offsetRange.second);

		RequestPair({ m_currentIndex, m_currentIndex + m_pairOffset });
	}

	void Controller::SetView(app::ViewSlot slot, const std::string& id)
	{
		const auto view = ViewTypeFromId(id);

		switch (slot)
		{
		case app::ViewSlot::View1:
			m_views.view1 = view;
			break;
		case app::ViewSlot::View2:
			m_views.view2 = view;
			break;
		}

		if (m_playState == PlayState::Pause && !m_cache.empty()) {
			RequestView(slot, view);
		}
	}

	void Controller::SetPairOffset(int pairOffset)
	{
		if (m_cache.empty())
			return;

		auto range = GetOffsetRange();
		m_pairOffset = std::clamp(pairOffset, range.first, range.second);

		RequestPair({ m_currentIndex, m_currentIndex + m_pairOffset });
	}

	std::pair<int, int> Controller::GetFramesRange() const
	{
		return std::make_pair(0, int(m_loader->NumImages()) - 2);
	}

	std::pair<int, int> Controller::GetOffsetRange() const
	{
		if (m_cache.empty())
			return {};

		int lastFrame = int(m_loader->NumImages()) - 1;
		int maxOffset = lastFrame - m_currentIndex;

		return { detail::minOffset, maxOffset };
	}

	int Controller::GetCurrentIndex() const
	{
		return m_currentIndex;
	}

	int Controller::GetPairOffset() const
	{
		return m_pairOffset;
	}

	std::vector<app::ViewOption> Controller::AllViewOptions() const
	{
		const auto allViews = cuda::motion::render::AllViews();

		std::vector<app::ViewOption> output;
		std::transform(allViews.begin(), allViews.end(), std::back_inserter(output), [](ViewType type) {
			return ViewTypeToViewOption(type);
		});

		return output;
	}

	void Controller::OnFrameReady(int index, size_t generation, QImage&& img)
	{
		if (generation != m_generation)
			return;

		m_cache[index] = std::move(img);
		TryAnalyseImagePair();
	}

	void Controller::TryAnalyseImagePair()
	{
		if (m_cache[m_requestedPair.first].isNull() || m_cache[m_requestedPair.second].isNull())
			return;

		if (m_requestPairSubmitted)
			return;

		auto jobInput = motion::AnalyseInput {
			m_requestedPair,
			m_generation,
			m_cache[m_requestedPair.first],
			m_cache[m_requestedPair.second],
			{ m_views.view1, m_views.view2 }
		};

		auto job = motion::CreateAnalyzeAndRenderJob(jobInput, [this](motion::AnalyseResult&& result) mutable {
			QMetaObject::invokeMethod(
				this, [this, res = std::move(result)]() mutable {
					OnGpuAnalysisResultReady(std::move(res));
				},
				Qt::QueuedConnection
			);
		});

		m_requestPairSubmitted = true;
		m_gpuWorker->AddJob(std::move(job));
	}

	void Controller::RequestView(ViewSlot slot, ViewType view)
	{
		if (m_displayedPair.first < 0 || m_displayedPair.second < 0)
			return;

		const auto requestedView = motion::RequestedView{ slot, view };
		auto jobInput = app::motion::RenderViewsInput {
			m_displayedPair,
			m_generation,
			{ requestedView }
		};

		auto job = motion::CreateRenderViewsJob(jobInput, [this](motion::RenderViewsResult&& result) mutable {
			QMetaObject::invokeMethod(
				this, [this, res = std::move(result)]() mutable {
					OnGpuRenderedViewReady(std::move(res));
				},
				Qt::QueuedConnection
			);
		});

		m_gpuWorker->AddJob(std::move(job));
	}

	void Controller::OnGpuRenderedViewReady(motion::RenderViewsResult&& result)
	{
		if (result.generation != m_generation)
			return;

		if (result.framePair != m_displayedPair)
			return;

		emit RenderedViewReady(std::move(result.views));
	}

	void Controller::OnDebugStats(IStatsProvider::Ptr statsProvider)
	{
		m_statsProvider = std::move(statsProvider);
		emit DebugStatsReady(*m_statsProvider);
	}

	void Controller::OnGpuAnalysisResultReady(motion::AnalyseResult&& result)
	{
		if (result.generation != m_generation)
			return;

		if (result.framePair != m_requestedPair)
			return;

		m_displayedPair = result.framePair;

		emit ImagesReady(
			result.prev,
			result.curr,
			result.views[m_views.view1],
			result.views[m_views.view2]
		);
	}
}
