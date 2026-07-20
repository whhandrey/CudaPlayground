#include "Shift.h"
#include "../KernelCommon.h"

#include <Cuda/MathUtils.h>
#include <Cuda/TimedCudaCall.h>

__global__ void ShiftImageKernel(
    const uchar4* __restrict__ image,
    size_t imgPitch,
    uchar4* __restrict__ output,
    size_t outputPitch,
    int width,
    int height,
    image::vec2i shiftVector)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    image::vec2i shifted = { x - shiftVector.x, y - shiftVector.y };
    bool valid = (shifted.x >= 0 && shifted.x < width) && (shifted.y >= 0 && shifted.y < height);

    uchar4 out_sample = {};

    if (valid) {
        const uchar4* inputRow = (const uchar4*)((const char*)image + shifted.y * imgPitch);
        out_sample = inputRow[shifted.x];
    }

    uchar4* outputRow = (uchar4*)((char*)output + y * outputPitch);
    outputRow[x] = out_sample;
}

namespace cuda {
    namespace transform {
        void ShiftImage(
            const GpuImageView<uchar4>& image,
            GpuImageView<uchar4>& output,
            image::vec2i shiftVector,
            cuda::KernelContext& ctx,
            image::vec2ui blockSize)
        {
            dim3 gridSize = cuda::math::Div(image.m_dim, blockSize);

            cuda::TimedCall("ShiftImageKernel", ctx, [&]() {
                ShiftImageKernel << <gridSize, cuda::math::vec2Todim3(blockSize), 0, ctx.m_stream >> > (
                    image.m_ptr,
                    image.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    image.m_dim.x,
                    image.m_dim.y,
                    shiftVector
                );
            });
        }
    }
}
