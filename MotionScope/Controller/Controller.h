#pragma once
#include <QImage>
#include <QObject>
#include <GpuProcessor/ViewRenderer/ViewType.h>
#include "../DisplayTypes/DisplayTypes.h"

namespace pool {
	class ThreadPool;
}

namespace image {
	class Loader;
}

namespace gpu {
	namespace motion {
		class AsyncGpuWorker;
		struct AnalyseResult;
		struct RenderViewsResult;
	}
}

namespace app {
	using cuda::motion::render::ViewType;

	class Controller : public QObject {
		Q_OBJECT

	public:
		Controller(std::unique_ptr<gpu::motion::AsyncGpuWorker> gpuWorker);
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
		void OnGpuAnalysisResultReady(gpu::motion::AnalyseResult&& result);

		void RequestView(ViewSlot slot, ViewType view);
		void OnGpuRenderedViewReady(gpu::motion::RenderViewsResult&& result);

	signals:
		void ImagesReady(QImage prev, QImage curr, QImage conf, QImage vis);
		void RenderedViewReady(std::map<ViewSlot, QImage> views);

	private:
		struct DisplayViews {
			ViewType view1;
			ViewType view2;
		};

	private:
		std::unique_ptr<image::Loader> m_loader;
		std::unique_ptr<pool::ThreadPool> m_pool;

		std::unique_ptr<gpu::motion::AsyncGpuWorker> m_gpuWorker;

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
