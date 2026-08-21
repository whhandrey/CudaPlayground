#include "Visualization.h"
#include "../../KernelCommon.h"
#include "../../VecUtils.h"

#include "../../Common.cuh"
#include "../../Vector.cuh"
#include "BlockStatsAggregation.cuh"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <math_constants.h>

using cuda::motion::BlockMatchStats;

__device__ __forceinline__ float2 ProjectPointOnLineSegment(float2 a, float2 b, float2 p) {
    float2 ab = Sub(b, a);
    float2 ap = Sub(p, a);

    float abLenSq = Dot(ab, ab);

    if (abLenSq < 1e-7f) {
        return a;
    }

    float t = clamp(Dot(ap, ab) / abLenSq, 0.0f, 1.0f);
    return Add(a, MulByConst(ab, t));
}

__device__ __forceinline__ float DistanceToSegment(float2 pixelCenter, float2 begin, float2 end) {
    // projected point p onto vector end-begin
    float2 x = ProjectPointOnLineSegment(begin, end, pixelCenter);

    float2 px = Sub(x, pixelCenter);

    return sqrtf(Dot(px, px));
}

__device__ __forceinline__ uchar4 ArrowColor(float2 vec) {
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

struct BoundingBox {
    int minX;
    int minY;
    int maxX;
    int maxY;
};

__device__ __forceinline__ BoundingBox MakeBoundingBox(
    float2 begin,
    float2 end,
    float thickness,
    int imageWidth,
    int imageHeight)
{
    const float radius = thickness * 0.5f;
    const float supportRadius = radius + 0.5f; // anti-aliasing margin

    BoundingBox box;

    // ceil is needed here cause it rounds towards positive infinity
    // in practive towards the next positive integer. ceil(33.2) == 34.
    box.minX = max(0, int(fminf(begin.x, end.x) - supportRadius));
    box.maxX = min(imageWidth - 1, int(ceilf(fmaxf(begin.x, end.x) + supportRadius)));

    box.minY = max(0, int(fminf(begin.y, end.y) - supportRadius));
    box.maxY = min(imageHeight - 1, int(ceilf(fmaxf(begin.y, end.y) + supportRadius)));

    return box;
}

__device__ __forceinline__ bool PtInsideBoundingBox(const BoundingBox& box, int x, int y) {
    return x >= box.minX
        && x <= box.maxX
        && y >= box.minY
        && y <= box.maxY;
}

__device__ __forceinline__ BoundingBox UnionBoundingBoxes(const BoundingBox& box1, const BoundingBox& box2) {
    BoundingBox boxOut;

    boxOut.minX = min(box1.minX, box2.minX);
    boxOut.maxX = max(box1.maxX, box2.maxX);
    boxOut.minY = min(box1.minY, box2.minY);
    boxOut.maxY = max(box1.maxY, box2.maxY);

    return boxOut;
}

__device__ __forceinline__ float CalculateCoverage(float2 begin, float2 end, float2 pt, float radius) {
    const float2 pixelCenter = make_float2(pt.x + 0.5f, pt.y + 0.5f);

    // distance from the pixelCenter to the center of the line (beg; end)
    // it has zero at the center of the segment (think of like zero of the coordinate system)
    const float distance = DistanceToSegment(pixelCenter, begin, end);

    // here zero moved to the boundary of the line
    const float signedDistance = distance - radius;

    // 0.5f is half of the pixel width
    // To account that pixel might be half inside, half outside, thus coverage should be 0.5f
    const float coverage = saturate(0.5f - signedDistance);
    return coverage;
}

__device__ __forceinline__ uchar4 Blend(uchar4 background, uchar4 foreground, float coverage) {
    uchar4 result;

    result.x = unsigned char(__float2int_rn(lerp(float(background.x), float(foreground.x), coverage)));

    result.y = unsigned char(__float2int_rn(lerp(float(background.y), float(foreground.y), coverage)));

    result.z = unsigned char(__float2int_rn(lerp(float(background.z), float(foreground.z), coverage)));

    result.w = 255;
    return result;
}

__device__ __forceinline__ void RasterizeSegmentCoverage(
    float* coverageOutput,
    size_t coveragePitch,
    int imageWidth,
    int imageHeight,
    float2 begin,
    float2 end,
    float thickness)
{
    const float2 segment = Sub(end, begin);
    if (Dot(segment, segment) < 1e-7f) {
        return;
    }

    BoundingBox box = MakeBoundingBox(begin, end, thickness, imageWidth, imageHeight);
    const float radius = thickness * 0.5f;

    for (int y = box.minY; y <= box.maxY; ++y) {
        float* row = (float*)((char*)coverageOutput + y * coveragePitch);

        for (int x = box.minX; x <= box.maxX; ++x) {
            const float2 pixelCenter = make_float2(float(x) + 0.5f, float(y) + 0.5f);

            // distance from the pixelCenter to the center of the line (beg; end)
            // it has zero at the center of the segment (think of like zero of the coordinate system)
            const float distance = DistanceToSegment(pixelCenter, begin, end);

            // here zero moved to the boundary of the line
            const float signedDistance = distance - radius;

            // 0.5f is half of the pixel width
            // To account that pixel might be half inside, half outside, thus coverage should be 0.5f
            const float coverage = saturate(0.5f - signedDistance);

            // Union with coverage already written by another segment of the same arrow.
            // In case if several threads write coverage to the same output.
            // This occurs cause we use three bounding boxes for each arrow's segment (shaft, left and right wings).
            row[x] = fmaxf(row[x], coverage);
        }
    }
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
    int arrowX = blockIdx.x * blockDim.x + threadIdx.x;
    int arrowY = blockIdx.y * blockDim.y + threadIdx.y;

    // divUp
    int arrowCountX = (statsDim.x + groupSize - 1) / groupSize;
    int arrowCountY = (statsDim.y + groupSize - 1) / groupSize;

    if (arrowX >= arrowCountX || arrowY >= arrowCountY) {
        return;
    }

    // aggregated shaft vector of neighboring bestDxDy vecs.
    const float2 agg_vec = AggregateVectorsInGroup(
        allStats,
        statsPitch,
        statsDim.x,
        statsDim.y,
        arrowX * groupSize,
        arrowY * groupSize,
        groupSize
    );

    const float lengthSqAgg = agg_vec.x * agg_vec.x + agg_vec.y * agg_vec.y;
    if (lengthSqAgg < 1e-10f) {
        return;
    }

    const float2 blockBegin = make_float2(arrowX * renderDim.x, arrowY * renderDim.y);

    const float radius = thickness * 0.5f;
    const float margin = radius + 0.5f; // 0.5f is // anti-aliasing margin

    // half of the width/height of the renderDim
    const float renderCellSize = float(min(renderDim.x, renderDim.y)) * 0.5f;
    const float2 begin = { blockBegin.x + renderCellSize, blockBegin.y + renderCellSize };

    const float invLengthAgg = rsqrtf(lengthSqAgg);
    const float2 direction = MulByConst(agg_vec, invLengthAgg);

    const float shaftLength = renderCellSize - margin;
    if (shaftLength < 1e-7f) {
        return;
    }

    const float2 tip = Add(begin, MulByConst(direction, shaftLength));

    // wings
    const float2 perpendicular = { -direction.y, direction.x };
    const float headLength = shaftLength * 0.4f;

    const float2 tip_float = make_float2(tip.x, tip.y);
    const float2 headBase = Sub(tip_float, MulByConst(direction, headLength));

    const float headAngle = 35.0f * CUDART_PI_F / 180.0f;
    const float headHalfWidth = headLength * tanf(headAngle);

    const float2 headLeft = Add(headBase, MulByConst(perpendicular, headHalfWidth));
    const float2 headRight = Sub(headBase, MulByConst(perpendicular, headHalfWidth));

    // render
    const uchar4 color = ArrowColor(agg_vec);
    const uchar4 background = make_uchar4(0, 0, 0, 0);

    const BoundingBox shaftBox = MakeBoundingBox(begin, tip, thickness, outDim.x, outDim.y);
    const BoundingBox leftWingBox = MakeBoundingBox(headLeft, tip, thickness, outDim.x, outDim.y);
    const BoundingBox rightWingBox = MakeBoundingBox(headRight, tip, thickness, outDim.x, outDim.y);

    const BoundingBox arrowBox = UnionBoundingBoxes(shaftBox, UnionBoundingBoxes(leftWingBox, rightWingBox));

    for (int y = arrowBox.minY; y <= arrowBox.maxY; ++y) {

        for (int x = arrowBox.minX; x <= arrowBox.maxX; ++x) {

            float coverage = 0.0f;
            if (PtInsideBoundingBox(shaftBox, x, y)) {
                coverage = max(coverage, CalculateCoverage(begin, tip, make_float2(x, y), radius));
            }

            if (PtInsideBoundingBox(leftWingBox, x, y)) {
                coverage = max(coverage, CalculateCoverage(headLeft, tip, make_float2(x, y), radius));
            }

            if (PtInsideBoundingBox(rightWingBox, x, y)) {
                coverage = max(coverage, CalculateCoverage(headRight, tip, make_float2(x, y), radius));
            }

            if (coverage > 0.0f) {
                uchar4* row = (uchar4*)((char*)output + y * outPitch);
                row[x] = Blend(background, color, saturate(coverage));
            }
        }
    }
}

namespace cuda {
    namespace motion {
        namespace visualization {

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
                    ArrowsMapKernel <<<gridSize, cuda::math::vec2Todim3(blockDim), 0, ctx.stream>>> (
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
