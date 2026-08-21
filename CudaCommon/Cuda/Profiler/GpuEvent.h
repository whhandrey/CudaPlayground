#pragma once
#include <string>
#include <cuda_runtime.h>

namespace cuda {
	namespace profile {
		class GpuEvent {
		public:
			GpuEvent();
			~GpuEvent() noexcept;

			GpuEvent(const GpuEvent&) = delete;
			GpuEvent& operator=(const GpuEvent&) = delete;

			GpuEvent(GpuEvent&& other) noexcept;
			GpuEvent& operator=(GpuEvent&& other) noexcept;

			cudaEvent_t Get() const noexcept;

		private:
			cudaEvent_t m_event = nullptr;
		};
	}
}
