#pragma once
#include <QImage>
#include <QObject>
#include <GpuProcessor/ViewRenderer/ViewType.h>
#include <Motion/Debug/Stats.h>
#include "../DisplayTypes/DisplayTypes.h"
#include "../Stats/StatsProvider.h"

namespace pool {
	class ThreadPool;
}

namespace image {
	class Loader;
}

namespace app::worker {
	class AsyncGpuWorker;
}

namespace app::motion {
	struct AnalyseResult;
	struct RenderViewsResult;
}

namespace app {
	using cuda::motion::render::ViewType;
	using ::motion::debug::StatsPacket;

	class StatsState;

	class Controller : public QObject {
		Q_OBJECT

	public:
		Controller(std::unique_ptr<app::worker::AsyncGpuWorker> gpuWorker);
		~Controller();

		void SetFolder(const std::string& folderPath);

		void SetPlayMode(PlayMode mode);
		void SetPlayState(PlayState state);

		bool TryStepForward();
		bool TryStepBackward();

		void PreparePlaybackStart();

		void SetFrame(int index);
		void SetView(ViewSlot slot, const std::string& id);
		void SetPairOffset(int pairOffset);

		std::pair<int, int> GetFramesRange() const;
		std::pair<int, int> GetOffsetRange() const;

		int GetCurrentIndex() const;
		int GetPairOffset() const;

		std::vector<ViewOption> AllViewOptions() const;

	private:
		void RequestPair(app::FramePair reqPair);
		void RequestFrame(int index);
		void OnFrameReady(int index, size_t generation, QImage&& img);
		void TryAnalyseImagePair();
		void OnGpuAnalysisResultReady(app::motion::AnalyseResult&& result);

		void RequestView(ViewSlot slot, ViewType view);
		void OnGpuRenderedViewReady(app::motion::RenderViewsResult&& result);

	signals:
		void ImagesReady(QImage prev, QImage curr, QImage conf, QImage vis);
		void RenderedViewReady(SlottedImage views);
		void DebugStatsReady(StatsProvider provider);

	private:
		struct DisplayViews {
			ViewType view1;
			ViewType view2;
		};

	private:
		std::unique_ptr<image::Loader> m_loader;
		std::unique_ptr<pool::ThreadPool> m_pool;

		std::unique_ptr<app::worker::AsyncGpuWorker> m_gpuWorker;
		std::unique_ptr<app::StatsState> m_statsState;

		size_t m_generation = 0;

		int m_currentIndex = 0;
		int m_pairOffset = 1;

		app::FramePair m_requestedPair = {};
		app::FramePair m_displayedPair = { -1, -1 };

		bool m_requestPairSubmitted = false;

		PlayMode m_playMode = PlayMode::Normal;
		PlayState m_playState = PlayState::Pause;

		DisplayViews m_views = {};

		std::vector<QImage> m_cache;
	};
}
