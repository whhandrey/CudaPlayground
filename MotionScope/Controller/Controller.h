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
	class IMotionGpuProcessor;
}

using GpuProcessor = std::unique_ptr<gpu::IMotionGpuProcessor>;

class Controller : public QObject {
	Q_OBJECT

public:
	Controller(GpuProcessor&& gpuProcessor);
	~Controller();

	void SetFolder(const std::string& folderPath);
	void RequestFrame(int index);

private:
	void OnFrameReady(int index, size_t generation, QImage&& image);
	void TryProcessImagePair();
	void OnGpuResultReady(QImage&& confImage);

private:
	std::unique_ptr<image::Loader> m_loader;
	std::unique_ptr<loader::ThreadPool> m_pool;

	GpuProcessor m_gpuProcessor;

	size_t m_generation = 0;
	size_t m_currentIndex = 0;

	std::vector<QImage> m_cache;
	bool m_gpuRunning = false;
};
