#pragma once
#include <QImage>
#include <QObject>
#include <GpuProcessor/ViewRenderer/ViewType.h>

namespace loader {
	class ThreadPool;
}

namespace image {
	class Loader;
}

namespace gpu {
	namespace motion {
		struct Result;
		class AsyncGpuWorker;
	}
}

namespace GpuApp {
	using cuda::motion::render::ViewType;

	enum class PlayMode {
		Normal,
		Loop
	};

	struct ViewOption {
		std::string id;
		std::string label;
	};

	enum class ViewSlot {
		View1,
		View2
	};

	struct DisplayViews {
		ViewType view1;
		ViewType view2;
	};

	class Controller : public QObject {
		Q_OBJECT

	public:
		Controller(std::unique_ptr<gpu::motion::AsyncGpuWorker> gpuWorker);
		~Controller();

		void SetFolder(const std::string& folderPath);
		void SetPlayMode(PlayMode mode);

		bool TryStepForward();
		bool TryStepBackward();

		void PreparePlaybackStart();

		void SetFrame(int index);
		void SetView(ViewSlot slot, const std::string& id);

		std::pair<int, int> GetFramesRange() const;
		int GetCurrentIndex() const;

		std::vector<ViewOption> AllViewOptions() const;

	private:
		void RequestFrame(int index);
		void OnFrameReady(int index, size_t generation);
		void TryProcessImagePair();
		void OnGpuResultReady(gpu::motion::Result result);

	signals:
		void ImagesReady(QImage prev, QImage curr, QImage conf, QImage vis);

	private:
		std::unique_ptr<image::Loader> m_loader;
		std::unique_ptr<loader::ThreadPool> m_pool;

		std::unique_ptr<gpu::motion::AsyncGpuWorker> m_gpuWorker;

		size_t m_generation = 0;
		int m_currentIndex = 0;

		PlayMode m_playMode = PlayMode::Normal;
		DisplayViews m_views;

		std::vector<QImage> m_cache;
	};
}
