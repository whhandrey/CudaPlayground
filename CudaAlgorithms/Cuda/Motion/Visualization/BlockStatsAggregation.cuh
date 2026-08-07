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
