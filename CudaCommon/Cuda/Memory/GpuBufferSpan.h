#pragma once
#include <Memory/GpuSpan.h>
#include "GpuBuffer.h"

namespace cuda::gpu_buffer {
	template <class T>
	memory::GpuSpan<T> MakeSpan(GpuBuffer<T>& buf) {
		return {
			buf.Data(),
			buf.Size()
		};
	}

	template <class T>
	memory::GpuSpan<const T> MakeSpan(const GpuBuffer<T>& buf) {
		return {
			buf.Data(),
			buf.Size()
		};
	}
}
