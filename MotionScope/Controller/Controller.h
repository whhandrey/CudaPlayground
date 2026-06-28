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
	class Controller : public QObject {
		Q_OBJECT

	public:
		Controller(std::unique_ptr<gpu::motion::AsyncGpuWorker> gpuWorker);
		~Controller();

		void SetFolder(const std::string& folderPath);
		void RequestFrame(int index);

	private:
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
		size_t m_currentIndex = 0;

		std::vector<QImage> m_cache;
	};
}
