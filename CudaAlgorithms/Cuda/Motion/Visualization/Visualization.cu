#include "Visualization.h"
#include "../../KernelCommon.h"
#include "../../Common.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <math_constants.h>

using cuda::motion::BlockMatchStats;

__device__ uchar4 HsvToRgb(float h, float s, float v)
{
    h = h - floorf(h); // wrap to [0,1)

    float r = 0.0f, g = 0.0f, b = 0.0f;

    float hh = h * 6.0f;
    int i = int(floorf(hh));
    float f = hh - i;

    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }

    return make_uchar4(
        static_cast<unsigned char>(r * 255.0f + 0.5f),
        static_cast<unsigned char>(g * 255.0f + 0.5f),
        static_cast<unsigned char>(b * 255.0f + 0.5f),
        255
    );
}

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

__global__  void MagnitudeMapKernel(
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

    float hue = 0.0f;
    float sat = 0.0f;
    float bri = 0.0f;

    const bool moved = (abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y)) > 0;

    if (moved) {
        float angle = atan2f(float(-stats->bestDxDy.y), float(stats->bestDxDy.x));
        hue = (angle + CUDART_PI_F) / (2.0f * CUDART_PI_F);

        float zeroScore = 0.0f;
        if (stats->zeroSad > 0) {
            zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
        }

        //float mag = sqrtf(float(stats->bestDxDy.x * stats->bestDxDy.x + stats->bestDxDy.y * stats->bestDxDy.y));
        //float normMag = saturate(mag / maxMag);

        sat = 1.0f; //saturate(zeroScore);
        bri = saturate(zeroScore);
    }

    uchar4 out_color = HsvToRgb(hue, sat, bri);

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

            void MagnitudeMap(
                const GpuImageView<BlockMatchStats>& stats,
                GpuImageView<uchar4>& output,
                cuda::KernelContext& ctx,
                image::vec2i search_halfsize,
                image::vec2ui blockDim)
            {
                dim3 gridSize = cuda::math::Div(stats.m_dim, blockDim);
                const float maxMag = sqrtf(search_halfsize.x * search_halfsize.x + search_halfsize.y * search_halfsize.y);

                cuda::TimedCall("MagnitudeMapKernel: " + cuda::util::BlockDimToString(blockDim), ctx, [&]() {
                    MagnitudeMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
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
