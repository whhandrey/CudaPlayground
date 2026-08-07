#pragma once
#include <cuda_runtime.h>
#include "../../Math.cuh"
#include "../BlockMatching.h"

using cuda::motion::BlockMatchStats;

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
            const BlockMatchStats* stats = rowStats + j;

            const bool moved = abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y) > 0;
            float conf = 0.0f;

            if (stats->zeroSad > 0) {
                float zeroScore = float(stats->zeroSad - stats->bestSad) / stats->zeroSad;
                conf = saturate(zeroScore) * float(moved);
            }

            if (conf > bestConf) {
                vec_out = { stats->bestDxDy.x, stats->bestDxDy.y };
                bestConf = conf;
            }
        }
    }

    return { float(vec_out.x), float(vec_out.y) };
}

__device__ __forceinline__ float StatsConfidence(const BlockMatchStats* stats) {
    const bool moved = abs(stats->bestDxDy.x) + abs(stats->bestDxDy.y) > 0;

    if (!moved || stats->zeroSad <= 0)
        return 0.0f;

    return saturate(float(stats->zeroSad - stats->bestSad) / float(stats->zeroSad));
}

__device__ __forceinline__ float2 AggregateVectorsInGroup(
    const BlockMatchStats* __restrict__ allStats,
    size_t pitch,
    int statsWidth,
    int statsHeight,
    int beginX,
    int beginY,
    int groupSize)
{
    const int endX = min(beginX + groupSize, statsWidth);
    const int endY = min(beginY + groupSize, statsHeight);

    float2 directionSum = {};
    float weightSum = 0.0f;

    // Pass 1: estimate the general direction.
    for (int y = beginY; y < endY; ++y) {
        const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + y * pitch);

        for (int x = beginX; x < endX; ++x) {
            const BlockMatchStats* stats = rowStats + x;

            float confidence = StatsConfidence(stats);
            if (confidence < 1e-7f)
                continue;

            float2 vec = make_float2(float(stats->bestDxDy.x), float(stats->bestDxDy.y));

            float lengthSq = Dot(vec, vec);
            if (lengthSq < 1e-7f)
                continue;

            // normalized
            float2 direction = MulByConst(vec, rsqrtf(lengthSq));

            directionSum.x += direction.x * confidence;
            directionSum.y += direction.y * confidence;

            weightSum += confidence;
        }
    }

    float directionSumLengthSq = Dot(directionSum, directionSum);
    if (directionSumLengthSq < 1e-7f || weightSum < 1e-7f)
        return make_float2(0.0f, 0.0f);

    float2 refDir = MulByConst(directionSum, rsqrtf(directionSumLengthSq));

    // Optional measure of how strongly the group agrees.
    const float coherence = sqrtf(directionSumLengthSq) / weightSum;

    if (coherence < 0.25f) {
        // No meaningful dominant motion.
        return make_float2(0.0f, 0.0f);
    }

    // cos(45 degrees). Accept vectors within ±45°.
    const float minDirAgreement = 0.70710678f;

    float2 filteredSum = {};

    // Pass 2: discard vectors pointing elsewhere.
    for (int y = beginY; y < endY; ++y) {
        const BlockMatchStats* rowStats = (BlockMatchStats*)((char*)allStats + y * pitch);

        for (int x = beginX; x < endX; ++x) {
            const BlockMatchStats* stats = rowStats + x;

            float confidence = StatsConfidence(stats);
            if (confidence <= 0.0f)
                continue;

            float2 vec = make_float2(float(stats->bestDxDy.x), float(stats->bestDxDy.y));

            float lengthSq = Dot(vec, vec);
            if (lengthSq < 1e-7f)
                continue;

            float2 direction = MulByConst(vec, rsqrtf(lengthSq));
            float agreement = Dot(direction, refDir);

            if (agreement < minDirAgreement)
                continue;

            filteredSum.x += direction.x * confidence;
            filteredSum.y += direction.y * confidence;
        }
    }

    return filteredSum;
}
