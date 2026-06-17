#include "AsyncGpuWorker.h"

namespace {
	ImageViewRGBA8 MakeImageView(const QImage& img) {
		return ImageViewRGBA8{
			img.constBits(),
			img.width(),
			img.height(),
			int(img.bytesPerLine())
		};
	}

	QImage ToQImage(ImageRGBA8&& image) {
		if (image.width <= 0 || image.height <= 0) {
			return {};
		}

		const int rowBytes = image.width * 4;
		const int pitch = image.pitchBytes > 0 ? image.pitchBytes : rowBytes;

		if (pitch < rowBytes) {
			throw std::logic_error("ToQImage: invalid pitch");
		}

		const auto requiredSize = static_cast<std::size_t>(pitch) * image.height;

		if (image.data.size() < requiredSize) {
			throw std::logic_error("ToQImage: not enough data");
		}

		// Need to alloc on heap to transfer ownership
		auto* buffer = new std::vector<unsigned char>(std::move(image.data));

		auto cleanup = [](void* info) {
			delete static_cast<std::vector<unsigned char>*>(info);
		};

		QImage qimg(
			buffer->data(),
			image.width,
			image.height,
			pitch,
			QImage::Format_RGBA8888,
			cleanup,
			buffer
		);

		return qimg;
	}
}

namespace gpu {
	namespace motion {
		AsyncGpuWorker::AsyncGpuWorker(std::unique_ptr<IMotionGpuProcessor> processor, ProcessCallback callback)
			: m_processor{ std::move(processor) }
			, m_callback{ std::move(callback) }
			, m_thread{ std::thread(&AsyncGpuWorker::Run, this) }
		{
		}

		AsyncGpuWorker::~AsyncGpuWorker() {
			Stop();
		}

		void AsyncGpuWorker::AddJob(const Job& job) {
			{
				std::lock_guard<std::mutex> lock(m_mtx);
				if (m_stop)
					return;

				m_jobs.push(job);
			}

			m_cv.notify_one();
		}

		void AsyncGpuWorker::Run() {
			while (true) {
				Job job;

				{
					std::unique_lock<std::mutex> lock(m_mtx);
					m_cv.wait(lock, [this]() { return m_stop || !m_jobs.empty(); });

					if (m_stop && m_jobs.empty())
						return;

					job = std::move(m_jobs.front());
					m_jobs.pop();
				}

				try {
					auto confImage = m_processor->Process(MakeImageView(job.m_prev), MakeImageView(job.m_curr));

					if (m_callback) {
						m_callback({ job.frameIndex, job.generation, job.m_prev, job.m_curr, ToQImage(std::move(confImage)) });
					}
				}
				catch (const std::exception&) {
					// TODO: later
				}
			}
		}

		void AsyncGpuWorker::Stop() {
			{
				std::lock_guard<std::mutex> lock(m_mtx);
				m_stop = true;
			}

			m_cv.notify_one();

			if (m_thread.joinable()) {
				m_thread.join();
			}

			// TODO: clear queue when move slider?
		}
	}
}