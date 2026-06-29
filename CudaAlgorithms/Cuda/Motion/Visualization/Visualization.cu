#include "Visualization.h"
#include "../../KernelCommon.h"
#include "../../Common.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

using cuda::motion::BlockMatchStats;

__global__  void ConfVisualizationKernel(
    const BlockMatchStats* __restrict__ allStats,
    size_t statsPitch,
    uchar4* __restrict__ output,
    size_t outPitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + y * statsPitch);
    const BlockMatchStats* stats = rowStats + x;

    float avg_sad = stats->count > 0 ? stats->sumSad / float(stats->count) : 0.0f;

    float conf = 0.0f;
    if (avg_sad > 0.0f) {
        conf = 1.0f - float(stats->bestSad) / avg_sad;
    }

    const int motion = abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y);

    float zeroScore = 0.0f;
    if (stats->zeroSad > 0) {
        zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
    }

    float moved = float(motion >= 1);
    float final_conf = (/*saturate(conf) * 0.5 + */saturate(zeroScore) * 0.5f) * moved;

    unsigned char out_sample = unsigned char(final_conf * 255.0f + 0.5f);

    uchar4* rowOut = (uchar4*)((char*)output + y * outPitch);
    rowOut[x] = make_uchar4(out_sample, out_sample, out_sample, 255);
}

namespace cuda {
    namespace motion {
        namespace visualization {
            void Conf(
                const GpuImageView<BlockMatchStats>& stats,
                GpuImageView<uchar4>& output,
                cuda::KernelContext& ctx,
                image::vec2ui blockDim)
            {
                dim3 gridSize = cuda::math::Div(stats.m_dim, blockDim);

                cuda::TimedCall("ConfKernel: " + cuda::util::BlockDimToString(blockDim), ctx, [&]() {
                    ConfVisualizationKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        stats.m_dim.x,
                        stats.m_dim.y
                    );
                });
            }
        }
    }
}
