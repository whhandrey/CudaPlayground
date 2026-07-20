#include "BlockMatchingWarp.h"
#include "BlockMatchingCommon.cuh"
#include "../KernelCommon.h"

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>
#include <cassert>

__device__ __forceinline__ int WarpReduceSum(int value) {
    // Include all lanes from the warp
    // 11111111 11111111 11111111 11111111
    // This ensures all lanes from the warp participate
    const unsigned int mask = 0xffffffffu;

    // 0 + 16, 1 + 16 ... 15 + 16 -> read all values and sum-up in the first half of the all lanes
    value += __shfl_down_sync(mask, value, 16);

    // 0 + 8, 1 + 8 ... 7 + 8 -> read first 1/4 of lanes
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);

    return value;
}

// Not in use now cause this version is way slower than simple block matching
__global__ void BlockMatchingWarpKernel(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    int4* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height,
    image::vec2i search_halfsize,
    image::vec2i macroBlockDim)
{
    extern __shared__ uchar4 tiles[];

    uchar4* prevTile = tiles;
    uchar4* currTile = tiles + (macroBlockDim.x * macroBlockDim.y);

    int baseBlockX = blockIdx.x * macroBlockDim.x;
    int baseBlockY = blockIdx.y * macroBlockDim.y;

    int blockSize = blockDim.x * blockDim.y;
    int macroBlockSize = macroBlockDim.x * macroBlockDim.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = tid; i < macroBlockSize; i += blockSize) {
        int localX = i % macroBlockDim.x;
        int localY = i / macroBlockDim.x;

        int globalX = baseBlockX + localX;
        int globalY = baseBlockY + localY;

        const uchar4* rowPrev = (const uchar4*)((const char*)prevFrame + globalY * prevPitch);
        prevTile[i] = rowPrev[globalX];
    }

    range dx_range = {
        max(-baseBlockX, -search_halfsize.x),
        min(search_halfsize.x, width - (baseBlockX + macroBlockDim.x))
    };

    range dy_range = {
        max(-baseBlockY, -search_halfsize.y),
        min(search_halfsize.y, height - (baseBlockY + macroBlockDim.y))
    };

    int currTileW = macroBlockDim.x + (dx_range.max - dx_range.min);
    int currTileH = macroBlockDim.y + (dy_range.max - dy_range.min);

    int currTileSize = currTileW * currTileH;

    for (int i = tid; i < currTileSize; i += blockSize) {
        int localX = i % currTileW;
        int localY = i / currTileW;

        int globalX = baseBlockX + localX + dx_range.min;
        int globalY = baseBlockY + localY + dy_range.min;

        const uchar4* rowCurr = (const uchar4*)((const char*)currFrame + globalY * currPitch);
        currTile[i] = rowCurr[globalX];
    }

    __syncthreads();

    const int laneId = tid & 31; // tid % 32
    const int warpId = tid >> 5; // tid / 32
    const int warpsPerBlock = blockSize >> 5; // blockSize / 32

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;
    SadCandidate bestSad = { INT_MAX, 0, 0 };

    const int warpSize = 32;

    for (int candidate = warpId; candidate < allCandidates; candidate += warpsPerBlock) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        // if dx/dy positive -> you shift to the right otherwise to the left
        int dxOffset = abs(dx_range.min) + dx;
        int dyOffset = abs(dy_range.min) + dy;

        int partialSad = 0;

        for (int i = laneId; i < macroBlockSize; i += warpSize) {
            int localX = i % macroBlockDim.x;
            int localY = i / macroBlockDim.x;

            int prevIdx = localX + localY * macroBlockDim.x;
            int currIdx = (localX + dxOffset) + (localY + dyOffset) * currTileW;

            int3 prevSample = make_int3(prevTile[prevIdx].x, prevTile[prevIdx].y, prevTile[prevIdx].z);
            int3 currSample = make_int3(currTile[currIdx].x, currTile[currIdx].y, currTile[currIdx].z);

            int diffR = abs(prevSample.x - currSample.x);
            int diffG = abs(prevSample.y - currSample.y);
            int diffB = abs(prevSample.z - currSample.z);

            partialSad += (diffR + diffB + diffG);
        }

        int sad = WarpReduceSum(partialSad);

        if (laneId == 0 && sad < bestSad.sad) {
            bestSad = { sad, dx, dy };
        }
    }

    int fullTileSize = (macroBlockDim.x + search_halfsize.x * 2) * (macroBlockDim.y + search_halfsize.y * 2);
    SadCandidate* sadTile = (SadCandidate*)(currTile + fullTileSize);

    if (laneId == 0) {
        sadTile[warpId] = bestSad;
    }

    __syncthreads();

    if (tid == 0) {
        SadCandidate finalMin = { INT_MAX, 0, 0 };

        for (int i = 0; i < warpsPerBlock; ++i) {
            finalMin = MinSad(finalMin, sadTile[i]);
        }

        int4* rowMotion = (int4*)((char*)motionImg + blockIdx.y * motionPitch);
        rowMotion[blockIdx.x] = { finalMin.dx, finalMin.dy, 0, 0 };
    }
}

namespace cuda {
    namespace motion {
        void BlockMatchingWarp(
            const GpuImageView<uchar4>& prevFrame,
            const GpuImageView<uchar4>& currFrame,
            GpuImageView<int4>& output,
            const BlockMatchingParams& p,
            cuda::KernelContext& ctx)
        {
            assert((p.blockDim.x * p.blockDim.y) % 32 == 0);

            dim3 gridSize = cuda::math::Div(prevFrame.m_dim, p.macroBlockDim);

            const int warpSize = 32;
            const int numWarpsPerBlock = (p.blockDim.x * p.blockDim.y) / warpSize;

            const size_t prevTileSize = (p.macroBlockDim.x * p.macroBlockDim.y) * sizeof(uchar4);
            const size_t currTileSize = (p.macroBlockDim.x + p.search_halfsize.x * 2) * (p.macroBlockDim.y + p.search_halfsize.y * 2) * sizeof(uchar4);
            const size_t sadTileSize = numWarpsPerBlock * sizeof(SadCandidate);

            const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
            image::vec2i macroBlockInt = { int(p.macroBlockDim.x), int(p.macroBlockDim.y) };

            cuda::TimedCall("BlockMatchingWarpKernel", ctx, [&]() {
                BlockMatchingWarpKernel<<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                    prevFrame.m_ptr,
                    prevFrame.m_pitch,
                    currFrame.m_ptr,
                    currFrame.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    prevFrame.m_dim.x,
                    prevFrame.m_dim.y,
                    p.search_halfsize,
                    macroBlockInt
                );
            });
        }
    }
}
