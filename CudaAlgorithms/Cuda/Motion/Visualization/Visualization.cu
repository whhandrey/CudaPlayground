#include "Visualization.h"
#include "../../KernelCommon.h"

#include "../../Common.cuh"
#include "../../Vector.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <math_constants.h>

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

    //float avg_sad = stats->count > 0 ? stats->sumSad / float(stats->count) : 0.0f;

    //float conf = 0.0f;
    //if (avg_sad > 0.0f) {
    //    conf = 1.0f - float(stats->bestSad) / avg_sad;
    //}

    const int motion = abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y);

    float zeroScore = 0.0f;
    if (stats->zeroSad > 0) {
        zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
    }

    float moved = float(motion >= 1);

    // this tried to combine uniqueness with zeroScore (did we move or not)
    float final_conf = (/*saturate(conf) * 0.5 + */saturate(zeroScore) * 0.5f) * moved;

    unsigned char out_sample = unsigned char(final_conf * 255.0f + 0.5f);

    uchar4* rowOut = (uchar4*)((char*)output + y * outPitch);
    rowOut[x] = make_uchar4(out_sample, out_sample, out_sample, 255);
}

__global__  void MotionMapKernel(
    const BlockMatchStats* __restrict__ allStats,
    size_t statsPitch,
    uchar4* __restrict__ output,
    size_t outPitch,
    int width,
    int height,
    float maxMag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + y * statsPitch);
    const BlockMatchStats* stats = rowStats + x;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float strength = 0.0f;

    const bool moved = (abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y)) > 0;

    float conf = 0.0f;
    if (stats->zeroSad > 0) {
        float zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
        conf = saturate(zeroScore);
    }

    if (moved) {
        float maxAxis = float(max(abs(stats->bestDxDy.x), abs(stats->bestDxDy.y)));
        if (maxAxis > 0.0f) {
            r = float(abs(stats->bestDxDy.x) > 0.0f);
            g = float(abs(stats->bestDxDy.y) > 0.0f);
        
            //r = abs(stats->bestDxDy.x) / maxAxis;
            //g = abs(stats->bestDxDy.y) / maxAxis;
        }

        //float mag = sqrtf(float(stats->bestDxDy.x * stats->bestDxDy.x + stats->bestDxDy.y * stats->bestDxDy.y));
        //float normMag = saturate(mag / maxMag);

        strength = /*normMag * */conf;
    }

    const float grey = 0.20f;

    uchar4 out_color = make_uchar4(
        normFloatToUchar(grey + (r - grey) * strength),
        normFloatToUchar(grey + (g - grey) * strength),
        normFloatToUchar(grey + (b - grey) * strength),
        255
    );

    uchar4* rowOut = (uchar4*)((char*)output + y * outPitch);
    rowOut[x] = out_color;
}

__global__  void MagMapKernel(
    const BlockMatchStats* __restrict__ allStats,
    size_t statsPitch,
    uchar4* __restrict__ output,
    size_t outPitch,
    int width,
    int height,
    float maxMag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + y * statsPitch);
    const BlockMatchStats* stats = rowStats + x;

    const bool moved = (abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y)) > 0;

    float conf = 0.0f;
    if (stats->zeroSad > 0) {
        float zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
        conf = saturate(zeroScore);
    }

    float out = 0.0f;
    if (moved) {
        float mag = sqrtf(stats->bestDxDy.x * stats->bestDxDy.x + stats->bestDxDy.y * stats->bestDxDy.y);
        out = saturate(conf * mag / maxMag);
    }

    unsigned char green_part = static_cast<unsigned char>(out * 255.0f + 0.5f);
    uchar4 out_color = make_uchar4(0, green_part, 0, 255);

    uchar4* rowOut = (uchar4*)((char*)output + y * outPitch);
    rowOut[x] = out_color;
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

                cuda::TimedCall("ConfKernel", ctx, [&]() {
                    ConfVisualizationKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        stats.m_dim.x,
                        stats.m_dim.y
                    );
                });
            }

            void MotionMap(
                const GpuImageView<BlockMatchStats>& stats,
                GpuImageView<uchar4>& output,
                cuda::KernelContext& ctx,
                image::vec2i search_halfsize,
                image::vec2ui blockDim)
            {
                dim3 gridSize = cuda::math::Div(stats.m_dim, blockDim);

                float maxMag = sqrtf(float(search_halfsize.x * search_halfsize.x + search_halfsize.y * search_halfsize.y));
                maxMag = maxMag < 1e-5f ? 1.0f : maxMag;

                cuda::TimedCall("MotionMapKernel", ctx, [&]() {
                    MotionMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        stats.m_dim.x,
                        stats.m_dim.y,
                        maxMag
                    );
                });
            }

            void MagMap(
                const GpuImageView<BlockMatchStats>& stats,
                GpuImageView<uchar4>& output,
                cuda::KernelContext& ctx,
                image::vec2i search_halfsize,
                image::vec2ui blockDim)
            {
                dim3 gridSize = cuda::math::Div(stats.m_dim, blockDim);

                float maxMag = sqrtf(float(search_halfsize.x * search_halfsize.x + search_halfsize.y * search_halfsize.y));
                maxMag = maxMag < 1e-5f ? 1.0f : maxMag;

                cuda::TimedCall("MagMapKernel", ctx, [&]() {
                    MagMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        stats.m_dim.x,
                        stats.m_dim.y,
                        maxMag
                    );
                });
            }
        }
    }
}
