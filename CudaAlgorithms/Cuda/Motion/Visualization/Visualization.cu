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

__device__ __forceinline__ float2 Mul(float2 a, float2 b) {
    return { a.x * b.x, a.y * b.y };
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

__device__ __forceinline__ bool PointNearLineSegment(float2 a, float2 b, float2 p, float thickness) {
    float2 ab = Sub(b, a);
    float abLenSq = Dot(ab, ab);

    if (abLenSq < 1e-7f) {
        return false;
    }

    // projected point p onto vector AB
    float2 x = ProjectPointOnLineSegment(a, b, p);
    float2 px = Sub(x, p);

    return Dot(px, px) < thickness * thickness;
}

__device__ __forceinline__ float DistanceToSegment(float2 pixelCenter, float2 begin, float2 end) {
    // projected point p onto vector end-begin
    float2 x = ProjectPointOnLineSegment(begin, end, pixelCenter);
    
    float2 px = Sub(x, pixelCenter);

    return sqrtf(Dot(px, px));
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
__device__ __forceinline__ float2 SelectBestDxDyVectorInGroup(
    const BlockMatchStats* __restrict__ allStats,
    size_t pitch,
    int statsWidth,
    int statsHeight,
    int beginX,
    int beginY,
    int groupSize)
{
    image::vec2i vec_out = {};

    int endX = min(beginX + groupSize, statsWidth);
    int endY = min(beginY + groupSize, statsHeight);

    float bestConf = 0.0f;
    for (int i = beginY; i < endY; ++i) {
        const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + i * pitch);

        for (int j = beginX; j < endX; ++j) {
            const BlockMatchStats& stats = rowStats[j];

            const bool moved = abs(stats.bestDxDy.x) + abs(stats.bestDxDy.y) > 0;
            float conf = 0.0f;

            if (stats.zeroSad > 0) {
                float zeroScore = float(stats.zeroSad - stats.bestSad) / stats.zeroSad;
                conf = saturate(zeroScore) * float(moved);
            }

            if (conf > bestConf) {
                vec_out = { stats.bestDxDy.x, stats.bestDxDy.y };
                bestConf = conf;
            }

            //vec_out.x += (stats.bestDxDy.x * conf);
            //vec_out.y += (stats.bestDxDy.y * conf);
        }
    }

    return { float(vec_out.x), float(vec_out.y) };
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

__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
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

// old one
__global__  void ArrowsMapKernelDDA(
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

    const uchar4 color = make_uchar4(64, 220, 255, 255);
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
    const float2 agg_vec = SelectBestDxDyVectorInGroup(
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
                row[x] = Blend(background, color, coverage);
            }
        }
    }
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
