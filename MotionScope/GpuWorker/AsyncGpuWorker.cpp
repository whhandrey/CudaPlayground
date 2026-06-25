#include "AsyncGpuWorker.h"
#include <Image/Image.h>

namespace {
	QImage ToQImage(image::Image<unsigned char>&& img) {
		if (img.Dim().x <= 0 || img.Dim().y <= 0) {
			return {};
		}

		const unsigned int rowBytes = img.Dim().x * sizeof(unsigned char);
		const size_t pitch = img.Pitch();

		if (pitch < rowBytes) {
			throw std::logic_error("ToQImage: invalid pitch");
		}

		// Need to alloc on heap to transfer ownership
		auto* buffer = new std::vector<unsigned char>(std::move(img.m_data));

		auto cleanup = [](void* info) {
			delete static_cast<std::vector<unsigned char>*>(info);
		};

		QImage qimg(
			buffer->data(),
			img.Dim().x,
			img.Dim().y,
			pitch,
			QImage::Format_Grayscale8,
			cleanup,
			buffer
		);

		return qimg;
	}

	image::ImageView<image::vec4uc> MakeImageView(const QImage& img) {
		if (img.format() != QImage::Format_RGBA8888) {
			throw std::logic_error("AsyncGpuWorker::MakeImageView: invalid input image, expected RGBA8888 format");
		}

		const image::vec4uc* ptr = reinterpret_cast<const image::vec4uc*>(img.constBits());

		assert(reinterpret_cast<std::uintptr_t>(ptr) % alignof(image::vec4uc) == 0);
		assert(img.bytesPerLine() % alignof(image::vec4uc) == 0);

		image::vec2ui dim = { static_cast<unsigned int>(img.width()), static_cast<unsigned int>(img.height()) };
		return image::ImageView<image::vec4uc> { ptr, dim, static_cast<size_t>(img.bytesPerLine()) };
	}
}

namespace gpu {
	namespace motion {
		AsyncGpuWorker::AsyncGpuWorker(std::unique_ptr<cuda::IMotionGpuProcessor> processor)
			: m_processor{ std::move(processor) }
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

		void AsyncGpuWorker::SetCallback(ProcessCallback&& callback) {
			m_callback = std::move(callback);
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
					auto result = Result {
						job.frameIndex,
						job.generation,
						job.m_prev,
						job.m_curr,
						ToQImage(std::move(confImage))
					};

					if (m_callback) {
						m_callback(std::move(result));
					}
				}
				catch (const std::exception& /*ex*/) {
					// m_exceptionMgr->Report(ex);
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