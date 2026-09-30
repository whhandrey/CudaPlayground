#pragma once

namespace memory {
	template <class T>
	struct GpuSpan {
		T* m_ptr;
		size_t m_size;
	};
}
