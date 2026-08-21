#include "GpuEvent.h"
#include "../CudaCheck.h"

namespace cuda::profile {
	GpuEvent::GpuEvent() {
		cudaCheck(cudaEventCreate(&m_event));
	}

	GpuEvent::~GpuEvent() noexcept {
		if (!m_event) {
			return;
		}

		// Report result through a nonthrowing logger.
		cudaEventDestroy(m_event);
	}

	GpuEvent::GpuEvent(GpuEvent&& other) noexcept
		: m_event{ std::exchange(other.m_event, nullptr) }
	{
	}

	GpuEvent& GpuEvent::operator=(GpuEvent&& other) noexcept {
		if (this == &other) {
			return *this;
		}

		if (m_event) {
			// TODO: report error?
			cudaEventDestroy(m_event);
		}

		m_event = std::exchange(other.m_event, nullptr);
		return *this;
	}

	cudaEvent_t GpuEvent::Get() const noexcept {
		return m_event;
	}
}