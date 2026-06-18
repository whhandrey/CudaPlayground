#pragma once
#include "Common.h"
#include "CudaTimer.h"
#include "Logger.h"
#include <functional>
#include <map>

namespace cuda {
	template <class Fn>
	void TimedCall(const std::string& name, cudaStream_t stream, Fn&& cudaKernel) {

		{
			CudaTimer timer(stream);
			
			cudaKernel();

			Logger().Log(name, timer.EndMs());
		}

		if (cudaGetLastError() != cudaSuccess) {
			Logger().Remove(name);
		}
	}
}
