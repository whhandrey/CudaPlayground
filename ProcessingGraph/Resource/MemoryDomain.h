#pragma once

namespace processing::resource {
	enum class MemoryDomain {
		Host,
		HostPinned,
		CudaDevice
	};
}
