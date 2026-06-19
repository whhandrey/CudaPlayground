#include "Reduction.cuh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

using uchar_t = image::uchar1;

struct SumCount {
    float sum;
    float count;
};

// Works for single channel, but can be extended
__global__ void AvgReductionKernel(
    const uchar_t* __restrict__ input,
    size_t inputPitch,
    float* __restrict__ output,
    size_t outputPitch,
    int width,
    int height,
    int2 tileSize)
{
    extern __shared__ SumCount tileBlock[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int localIdx = tx + ty * blockDim.x;
    const uchar_t* srcRow = (const uchar_t*)((const char*)input + y * inputPitch);

    const bool valid = x < width && y < height;
    tileBlock[localIdx] = valid ? SumCount{ float(srcRow[x]), 1.0f } : SumCount{};

    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        const int step = i * 2;

        // threadIdx.x % step && threadIdx.y % step, but works only if step is power of two
        const bool tileAnchor = ((tx & (step - 1)) == 0) && ((ty & (step - 1)) == 0);

        if (tileAnchor) {
            const int idx00 = tx + blockDim.x * ty;
            const int idx01 = (tx + i) + blockDim.x * ty;
            const int idx10 = tx + blockDim.x * (ty + i);
            const int idx11 = (tx + i) + blockDim.x * (ty + i);

            tileBlock[idx00].sum =
                tileBlock[idx00].sum +
                tileBlock[idx01].sum +
                tileBlock[idx10].sum +
                tileBlock[idx11].sum;

            tileBlock[idx00].count =
                tileBlock[idx00].count +
                tileBlock[idx01].count +
                tileBlock[idx10].count +
                tileBlock[idx11].count;
        }

        __syncthreads();
    }

    if (tx == 0 && ty == 0) {
        const float count = tileBlock[0].count;
        const float avg = count > 0 ? tileBlock[0].sum / count : 0.0f;

        float* dstRow = (float*)((char*)output + blockIdx.y * outputPitch);
        dstRow[blockIdx.x] = avg;
    }
}

namespace tile {
    namespace reduction {
        ImageGPU<float> Avg(const ImageGPU<uchar4>& input, image::vec2ui tileSize) {
            assert(tileSize.x == tileSize.y);
            assert((tileSize.x & (tileSize.x - 1)) == 0);

            dim3 gridSize = cuda::math::DivUp(input.Dim(), tileSize);

            ImageGPU<float> output(image::vec2ui{ gridSize.x, gridSize.y });
            return output;
        }
    }
}
