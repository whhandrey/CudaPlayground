#include "Visualization.h"
#include "../../KernelCommon.h"
#include "../../Common.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <math_constants.h>

using cuda::motion::BlockMatchStats;

namespace {
    template <class Vec2, class Vec1>
    Vec2 ConvertTo(Vec1 vec1) {
        return { vec1.x, vec1.y };
    }
}

__device__ __forceinline__ float Dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ __forceinline__ float2 Sub(float2 a, float2 b) {
    return { a.x - b.x, a.y - b.y };
}

__device__ __forceinline__ float2 Add(float2 a, float2 b) {
    return { a.x + b.x, a.y + b.y };
}

__device__ __forceinline__ float2 MulByConst(float2 a, float c) {
    return { a.x * c, a.y * c };
}

__device__ __forceinline__ float2 Normalize(float2 vec) {
    float len2 = vec.x * vec.x + vec.y * vec.y;

    if (len2 < 1e-10f) {
        return { 0, 0 };
    }

    float invLength = rsqrtf(vec.x * vec.x + vec.y * vec.y);
    return { vec.x * invLength, vec.y * invLength };
}

__device__ __forceinline__ void DrawBrush(
    uchar4* output,
    size_t pitch,
    int width,
    int height,
    int centerX,
    int centerY,
    float radius,
    uchar4 color)
{
    int extent = int(ceilf(radius));
    float radiusSq = radius * radius;

    for (int oy = -extent; oy <= extent; ++oy) {
        int y = centerY + oy;

        if (y < 0 || y >= height) {
            continue;
        }

        uchar4* row = (uchar4*)((char*)output + y * pitch);

        for (int ox = -extent; ox <= extent; ++ox) {

            // Circular brush, not a square one.
            if (float(ox * ox + oy * oy) > radiusSq)
                continue;

            int x = centerX + ox;

            if (x < 0 || x >= width)
                continue;

            row[x] = color;
        }
    }
}

__device__ __forceinline__ void DrawLineDDA(
    uchar4* output,
    size_t pitch,
    int width,
    int height,
    int2 begin,
    int2 end,
    float thickness,
    uchar4 color)
{
    int dx = end.x - begin.x;
    int dy = end.y - begin.y;

    // Number of pixel-sized steps along the longest direction.
    int steps = max(abs(dx), abs(dy));

    float radius = thickness * 0.5f;

    // no need to draw if zero movement yet
    if (steps == 0) {
        //DrawBrush(output, pitch, width, height, begin.x, begin.y, radius, color);
        return;
    }

    // Movement during one DDA iteration.
    float stepX = dx / float(steps);
    float stepY = dy / float(steps);

    float x = float(begin.x);
    float y = float(begin.y);

    // <= includes final end point
    for (int i = 0; i <= steps; ++i) {
        DrawBrush(output, pitch, width, height, __float2int_rn(x), __float2int_rn(y), radius, color);

        x += stepX;
        y += stepY;
    }
}

// NB: no averaging by the total weight, the vector will anyway be normalized
__device__ __forceinline__ float2 VecFromNeighbors(
    const BlockMatchStats* __restrict__ allStats,
    size_t pitch,
    int statsWidth,
    int statsHeight,
    int beginX,
    int beginY,
    int groupSize)
{
    float2 vec_out = { 0.0f, 0.0f };

    int endX = min(beginX + groupSize, statsWidth);
    int endY = min(beginY + groupSize, statsHeight);

    for (int i = beginY; i < endY; ++i) {
        const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + i * pitch);

        for (int j = beginX; j < endX; ++j) {
            const BlockMatchStats& stats = rowStats[j];

            bool moved = abs(stats.bestDxDy.x) + abs(stats.bestDxDy.y) > 0;
            if (!moved) {
                continue;
            }

            float conf = 0.0f;
            if (stats.zeroSad > 0) {
                float zeroScore = float(stats.zeroSad - stats.bestSad) / stats.zeroSad;
                conf = saturate(zeroScore) * float(moved);
            }

            vec_out.x += (stats.bestDxDy.x * conf);
            vec_out.y += (stats.bestDxDy.y * conf);
        }
    }

    return vec_out;
}

__device__ __forceinline__ uchar4 ArrowColor(image::vec2i vec) {
    if (vec.x == 0) {
        return make_uchar4(255, 50, 50, 255);
    }

    if (vec.y == 0) {
        return make_uchar4(50, 255, 50, 255);
    }

    return make_uchar4(64, 220, 255, 255);
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

__global__  void ArrowsMapKernel(
    const BlockMatchStats* __restrict__ allStats,
    size_t statsPitch,
    uchar4* __restrict__ output,
    size_t outPitch,
    uint2 statsDim,
    uint2 outDim,
    int2 renderDim,
    int groupSize,
    float thickness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // divUp
    int arrowCountX = (statsDim.x + groupSize - 1) / groupSize;
    int arrowCountY = (statsDim.y + groupSize - 1) / groupSize;

    if (x >= arrowCountX || y >= arrowCountY)
        return;

    // aggregated vec of neighboring bestDxDy vecs.
    float2 agg_vec = VecFromNeighbors(allStats, statsPitch, statsDim.x, statsDim.y, x * groupSize, y * groupSize, groupSize);

    float lengthSq = agg_vec.x * agg_vec.x + agg_vec.y * agg_vec.y;
    if (lengthSq < 1e-10f) {
        return;
    }

    int2 blockBegin = { x * renderDim.x, y * renderDim.y };
    int renderHeight = min(renderDim.x, renderDim.y);

    int2 begin = { blockBegin.x + int(renderHeight * 0.5f), blockBegin.y + int(renderHeight * 0.5f) };

    float invLength = rsqrtf(lengthSq);
    float2 vec_norm = { agg_vec.x * invLength, agg_vec.y * invLength };
    
    int2 end = {
        begin.x + int(vec_norm.x * renderHeight * 0.5f),
        begin.y + int(vec_norm.y * renderHeight * 0.5f)
    };

    const uchar4 color = make_uchar4(64, 220, 255, 255);
    DrawLineDDA(output, outPitch, outDim.x, outDim.y, begin, end, thickness, color);
}

namespace {
    template <class Vec1, class Vec2>
    bool EqVec(Vec1 a, Vec2 b) {
        return a.x == b.x && a.y == b.y;
    }
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
                    MotionMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
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
                    MagMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
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

            void ArrowsMap(
                const GpuImageView<BlockMatchStats>& stats,
                GpuImageView<uchar4>& output,
                cuda::KernelContext& ctx,
                image::vec2ui macroBlockDim,
                int groupSize,
                float thickness,
                image::vec2ui blockDim)
            {
                unsigned int arrowCountX = cuda::math::DivUp(stats.m_dim.x, unsigned int(groupSize));
                unsigned int arrowCountY = cuda::math::DivUp(stats.m_dim.y, unsigned int(groupSize));

                dim3 gridSize = cuda::math::DivUp({ arrowCountX, arrowCountY }, blockDim);
                int2 renderDim = { int(macroBlockDim.x) * groupSize, int(macroBlockDim.y) * groupSize };

                cuda::TimedCall("ArrowsMapKernel", ctx, [&]() {
                    ArrowsMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        ConvertTo<uint2>(stats.m_dim),
                        ConvertTo<uint2>(output.m_dim),
                        renderDim,
                        groupSize,
                        thickness
                    );
                });
            }
        }
    }
}
