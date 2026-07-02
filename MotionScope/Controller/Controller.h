#pragma once
#include <QImage>
#include <QObject>

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
	enum class PlayMode {
		Normal,
		Loop
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

		void SetFrame(int index);

		std::pair<int, int> GetFramesRange() const;
		int GetCurrentIndex() const;

	private:
		void RequestFrame(int index);
		void OnFrameReady(int index, size_t generation, QImage&& image);
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

		std::vector<QImage> m_cache;
	};
}
