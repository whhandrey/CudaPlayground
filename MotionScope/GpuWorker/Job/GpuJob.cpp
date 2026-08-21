#include "GpuJob.h"
#include "../../GpuProcessor/MotionViewProcessor.h"
#include <Image/ImageView.h>
#include <stdexcept>
#include <iterator>

namespace {
	using cuda::motion::render::ViewType;

	QImage ToQImage(image::Image<image::vec4uc>&& img) {
		if (img.Dim().x <= 0 || img.Dim().y <= 0) {
			return {};
		}

		const unsigned int rowBytes = img.Dim().x * sizeof(image::vec4uc);
		const size_t pitch = img.Pitch();

		if (pitch < rowBytes) {
			throw std::logic_error("ToQImage: invalid pitch");
		}

		// Need to alloc on heap to transfer ownership
		auto* buffer = new std::vector<image::vec4uc>(std::move(img.m_data));

		auto cleanup = [](void* info) {
			delete static_cast<std::vector<image::vec4uc>*>(info);
			};

		QImage qimg(
			reinterpret_cast<unsigned char*>(buffer->data()),
			img.Dim().x,
			img.Dim().y,
			pitch,
			QImage::Format_RGBA8888,
			cleanup,
			buffer
		);

		return qimg;
	}

	std::map<ViewType, QImage> ToQImages(std::map<ViewType, image::Image<image::vec4uc>>&& views) {
		std::map<ViewType, QImage> output;

		for (auto& img : views) {
			output.emplace(img.first, ToQImage(std::move(img.second)));
		}

		return output;
	}

	app::motion::VersionedFrame MakeView(const QImage& img, size_t index, size_t generation) {
		if (img.format() != QImage::Format_RGBA8888) {
			throw std::logic_error("AsyncGpuWorker::MakeImageView: invalid input image, expected RGBA8888 format");
		}

		const image::vec4uc* ptr = reinterpret_cast<const image::vec4uc*>(img.constBits());

		assert(reinterpret_cast<std::uintptr_t>(ptr) % alignof(image::vec4uc) == 0);
		assert(img.bytesPerLine() % alignof(image::vec4uc) == 0);

		image::vec2ui dim = { static_cast<unsigned int>(img.width()), static_cast<unsigned int>(img.height()) };
		auto imageView = image::CpuImageView<const image::vec4uc>{ ptr, dim, static_cast<size_t>(img.bytesPerLine()) };

		return { imageView, index, generation };
	}
}

namespace app::motion {
	class AnalyseRenderJobAsync : public IGpuJob {
	public:
		AnalyseRenderJobAsync(const AnalyseInput& input, AnalyseResultCb&& callback);

		void Execute(MotionViewProcessor& proc) override;

	private:
		AnalyseInput m_input;
		AnalyseResultCb m_callback;
	};

	class RenderViewsJobAsync : public IGpuJob {
	public:
		RenderViewsJobAsync(const RenderViewsInput& input, RenderViewsResultCb&& callback);

		void Execute(MotionViewProcessor& proc) override;

	private:
		RenderViewsInput m_input;
		RenderViewsResultCb m_callback;
	};

	AnalyseRenderJobAsync::AnalyseRenderJobAsync(const AnalyseInput& input, AnalyseResultCb&& callback)
		: m_input{ input }
		, m_callback{ std::move(callback) }
	{
	}

	void AnalyseRenderJobAsync::Execute(MotionViewProcessor& proc) {
		std::vector<ViewType> viewsRequest;
		const auto& request = m_input.requestedViews;

		std::transform(request.begin(), request.end(), std::back_inserter(viewsRequest), [](const auto& slotView) {
			return slotView.view;
		});

		auto views = proc.AnalyzeAndRenderViews(
			MakeView(m_input.prev, m_input.framePair.first, m_input.generation),
			MakeView(m_input.curr, m_input.framePair.second, m_input.generation),
			viewsRequest
		);

		auto images = ToQImages(std::move(views));

		std::vector<SlottedImage> slottedImages;

		for (const auto& requestedView : request) {
			const auto it = images.find(requestedView.view);

			if (it == images.end()) {
				throw std::logic_error("AnalyseRenderJobAsync::Execute: processor did not return a requested view");
			}

			slottedImages.emplace_back(SlottedImage{ requestedView.slot, it->second });
		}

		auto result = AnalyseResult {
			m_input.framePair,
			m_input.generation,
			m_input.prev,
			m_input.curr,
			std::move(slottedImages),
			proc.TakeLastStats()
		};

		if (m_callback) {
			m_callback(std::move(result));
		}
	}

	RenderViewsJobAsync::RenderViewsJobAsync(const RenderViewsInput& input, RenderViewsResultCb&& callback)
		: m_input{ input }
		, m_callback{ std::move(callback) }
	{
	}

	void RenderViewsJobAsync::Execute(MotionViewProcessor& proc)
	{
		auto typedView = ToQImages(proc.RenderViews({ m_input.requestedView.view }));

		SlottedImage output {
			m_input.requestedView.slot,
			std::move(typedView.at(m_input.requestedView.view))
		};

		auto result = RenderViewsResult {
			m_input.framePair,
			m_input.generation,
			std::move(output),
			proc.TakeLastStats()
		};

		if (m_callback) {
			m_callback(std::move(result));
		}
	}

	IGpuJob::Ptr CreateAnalyzeAndRenderJob(const AnalyseInput& input, AnalyseResultCb&& callback)
	{
		return std::make_unique<AnalyseRenderJobAsync>(input, std::move(callback));
	}

	IGpuJob::Ptr CreateRenderViewsJob(const RenderViewsInput& input, RenderViewsResultCb&& callback)
	{
		return std::make_unique<RenderViewsJobAsync>(input, std::move(callback));
	}
}
