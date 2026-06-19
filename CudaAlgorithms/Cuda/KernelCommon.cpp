#include "KernelCommon.h"
#include <Cuda/CudaCheck.h>

namespace cuda {
	namespace gpu {
		Arch DetectArch(int device)
		{
			cudaDeviceProp prop{};
			cudaCheck(cudaGetDeviceProperties(&prop, device));

			const int cc = prop.major * 10 + prop.minor;

			// Common mapping:
			// 7.5  -> Turing
			// 8.x  -> Ampere
			// 8.9  -> Ada
			// 12.x or similar -> newer Blackwell-class, depending CUDA version/device
			if (cc == 75)
				return Arch::Turing;

			if (prop.major == 8 && prop.minor != 9)
				return Arch::Ampere;

			if (prop.major == 8 && prop.minor == 9)
				return Arch::Ada;

			if (prop.major >= 9)
				return Arch::Blackwell;

			return Arch::Unknown;
		}
	}
}
