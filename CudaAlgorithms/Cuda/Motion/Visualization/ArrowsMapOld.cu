#include "Visualization.h"
#include "../../KernelCommon.h"
#include "../../VecUtils.h"

#include "../../Common.cuh"
#include "../../Vector.cuh"
#include "BlockStatsReduction.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <math_constants.h>

using cuda::motion::BlockMatchStats;

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

__device__ __forceinline__ uchar4 ArrowColor(float2 vec)
{
    float absX = fabsf(vec.x);
    float absY = fabsf(vec.y);

    if (absX >= absY) {
        // Mostly horizontal
        // right: red, left: cyan
        return vec.x >= 0.0f ? make_uchar4(255, 50, 50, 255) : make_uchar4(64, 220, 255, 255);
    }

    // Mostly vertical
    // down: green, up: yellow
    return vec.y >= 0.0f ? make_uchar4(50, 255, 50, 255) : make_uchar4(255, 220, 50, 255);
}

__global__  void ArrowsMapDDAKernel(
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

    // aggregated shaft vector of neighboring bestDxDy vecs.
    float2 agg_vec = SelectBestDxDyVectorInGroup(allStats, statsPitch, statsDim.x, statsDim.y, x * groupSize, y * groupSize, groupSize);

    float lengthSqAgg = agg_vec.x * agg_vec.x + agg_vec.y * agg_vec.y;
    if (lengthSqAgg < 1e-10f) {
        return;
    }

    int2 blockBegin = { x * renderDim.x, y * renderDim.y };
    const int renderHeight = min(renderDim.x, renderDim.y);

    const float shaftLength = renderHeight * 0.5f;
    int2 begin = { blockBegin.x + int(shaftLength), blockBegin.y + int(shaftLength) };

    float invLengthAgg = rsqrtf(lengthSqAgg);
    float2 direction = MulByConst(agg_vec, invLengthAgg);
    
    int2 tip = {
        begin.x + int(ceilf(direction.x * shaftLength)),
        begin.y + int(ceilf(direction.y * shaftLength))
    };

    const uchar4 color = ArrowColor(make_float2(agg_vec.x, agg_vec.y));
    DrawLineDDA(output, outPitch, outDim.x, outDim.y, begin, tip, thickness, color);

    // arrows
    float2 perpendicular = { -direction.y, direction.x };
    float headLength = shaftLength * 0.4f;

    float2 tip_float = make_float2(tip.x, tip.y);
    float2 headBase = Sub(tip_float, MulByConst(direction, headLength));

    float headAngle = 35.0f * CUDART_PI_F / 180.0f;
    const float headHalfWidth = headLength * tanf(headAngle);

    float2 headLeft = Add(headBase, MulByConst(perpendicular, headHalfWidth));
    float2 headRight = Sub(headBase, MulByConst(perpendicular, headHalfWidth));

    DrawLineDDA(output, outPitch, outDim.x, outDim.y, float2ToInt2(headLeft), tip, thickness, color);
    DrawLineDDA(output, outPitch, outDim.x, outDim.y, float2ToInt2(headRight), tip, thickness, color);
}

namespace cuda {
    namespace motion {
        namespace visualization {

            void ArrowsMapDDA(
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

                cuda::TimedCall("ArrowsMapDDAKernel", ctx, [&]() {
                    ArrowsMapDDAKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.m_stream>>> (
                        stats.m_ptr,
                        stats.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        ::util::vec::ConvertTo<uint2>(stats.m_dim),
                        ::util::vec::ConvertTo<uint2>(output.m_dim),
                        renderDim,
                        groupSize,
                        thickness
                    );
                });
            }
        }
    }
}
