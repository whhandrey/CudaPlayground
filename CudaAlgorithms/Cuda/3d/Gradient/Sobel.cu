#include "Sobel.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <map>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__  void SobelMagnitude3dKernel()
{
}

namespace cuda {
	namespace processing3d {
		void SobelMagnitude(
			const image::GpuImageView<uchar4>& input,
			image::GpuImageView<uchar4>& output,
			cuda::KernelContext& ctx
		)
		{

		}
	}
}
