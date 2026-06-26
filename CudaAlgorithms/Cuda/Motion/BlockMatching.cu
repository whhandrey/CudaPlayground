#include "BlockMatching.h"
#include "BlockMatchingCommon.cuh"
#include "../Common.cuh"
#include "../KernelCommon.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <map>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

using cuda::motion::BlockMatchStats;

#define PRINT_KERNEL_ATTRS(kernelName)                                       \
do {                                                                         \
    cudaFuncAttributes attr{};                                               \
    cudaCheck(cudaFuncGetAttributes(&attr, kernelName));                     \
    std::cout << #kernelName                                                 \
              << " binaryVersion=" << attr.binaryVersion                     \
              << " ptxVersion=" << attr.ptxVersion                           \
              << " numRegs=" << attr.numRegs                                 \
              << " sharedSizeBytes=" << attr.sharedSizeBytes                 \
              << " constSizeBytes=" << attr.constSizeBytes                   \
              << '\n';                                                       \
} while (0)

namespace {
    struct SadTop2 {
        SadCandidate best;
        SadCandidate second;
    };

    struct SadStats {
        SadTop2 top2;

        unsigned long sumSad;
        int zeroSad;
    };
}

__device__ __forceinline__ bool ValidCandidate(SadCandidate c) {
    return c.sad != INT_MAX;
}

__device__ __forceinline__ SadCandidate EmptySadCand() {
    return { INT_MAX, 0, 0 };
}

__device__ __forceinline__ SadTop2 EmptySadTop2() {
    return { EmptySadCand(), EmptySadCand() };
}

__device__ __forceinline__ SadStats EmptySadStats() {
    return { EmptySadTop2(), 0, 0 };
}

__device__ __forceinline__ void UpdateBestSad(SadStats& sadLocal, SadCandidate sadCand) {
    if (sadCand.sad < sadLocal.top2.best.sad) {
        sadLocal.top2.second = sadLocal.top2.best;
        sadLocal.top2.best = sadCand;
    }
    else if (sadCand.sad < sadLocal.top2.second.sad) {
        sadLocal.top2.second = sadCand;
    }

    sadLocal.sumSad += sadCand.sad;
    if (sadCand.dx == 0 && sadCand.dy == 0) {
        sadLocal.zeroSad = sadCand.sad;
    }
}

__device__ __forceinline__ SadTop2 MergeSadTop2(SadTop2 first, SadTop2 second) {
    if (!ValidCandidate(first.best)) {
        return second;
    }

    if (!ValidCandidate(second.best)) {
        return first;
    }

    bool firstSadLower = first.best.sad < second.best.sad;

    SadTop2 winner = firstSadLower ? first : second;
    SadTop2 notWinner = firstSadLower ? second : first;

    SadTop2 output;
    output.best = winner.best;
    output.second = MinSad(winner.second, notWinner.best);

    return output;
}

__global__  void BlockMatchingKernel(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    BlockMatchStats* __restrict__ statsOut,
    size_t statsOutPitch,
    int width,
    int height,
    image::vec2i search_halfsize,
    image::vec2i macroBlockDim)
{
    extern __shared__ uchar4 tiles[];

    uchar4* prevTile = tiles;
    uchar4* currTile = prevTile + (macroBlockDim.x * macroBlockDim.y);

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

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;

    // collect full info now for testing
    SadStats localStats = EmptySadStats();

    int currTileBaseX = abs(dx_range.min);
    int currTileBaseY = abs(dy_range.min);

    for (int candidate = tid; candidate < allCandidates; candidate += blockSize) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        int sad = 0.0f;
        for (int i = 0; i < macroBlockDim.y; i++) {
            uchar4* prev = prevTile + (i * macroBlockDim.x);
            uchar4* curr = currTile + (i + currTileBaseY + dy) * currTileW;

            for (int j = 0; j < macroBlockDim.x; ++j) {
                const int currX = j + currTileBaseX + dx;

                sad += abs(int(prev[j].x) - int(curr[currX].x))
                     + abs(int(prev[j].y) - int(curr[currX].y))
                     + abs(int(prev[j].z) - int(curr[currX].z));
            }
        }

        UpdateBestSad(localStats, { sad, dx, dy });
    }

    const int fullTileSize = (macroBlockDim.x + search_halfsize.x * 2) * (macroBlockDim.y + search_halfsize.y * 2);

    SadStats* sadTile = (SadStats*)(currTile + fullTileSize);

    bool valid = tid < allCandidates;
    sadTile[tid] = valid ? localStats : EmptySadStats();

    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SadStats curr = sadTile[tid];
            SadStats next = sadTile[tid + stride];

            SadTop2 out = MergeSadTop2(curr.top2, next.top2);

            unsigned long sumSad = curr.sumSad + next.sumSad;
            int zeroSad = curr.zeroSad + next.zeroSad;

            sadTile[tid] = SadStats{ out, sumSad, zeroSad };
        }

        __syncthreads();
    }

    if (tid == 0) {
        BlockMatchStats* rowStats = (BlockMatchStats*)((char*)statsOut + blockIdx.y * statsOutPitch);

        BlockMatchStats out {
            { sadTile[0].top2.best.dx, sadTile[0].top2.best.dy },
            { sadTile[0].top2.second.dx, sadTile[0].top2.second.dy },

            sadTile[0].top2.best.sad,
            sadTile[0].top2.second.sad,

            sadTile[0].zeroSad,
            sadTile[0].sumSad
        };

        rowStats[blockIdx.x] = out;
    }
}

template <
    int macroBlockW,
    int macroBlockH,
    int search_halfsizeX,
    int search_halfsizeY
>
__global__ void BlockMatchingKernelT(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    BlockMatchStats* __restrict__ statsOut,
    size_t statsOutPitch,
    int width,
    int height)
{
    extern __shared__ uchar4 tiles[];

    uchar4* prevTile = tiles;
    uchar4* currTile = prevTile + (macroBlockW * macroBlockH);

    int baseBlockX = blockIdx.x * macroBlockW;
    int baseBlockY = blockIdx.y * macroBlockH;

    int blockSize = blockDim.x * blockDim.y;
    constexpr int macroBlockSize = macroBlockW * macroBlockH;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    #pragma unroll 1
    for (int i = tid; i < macroBlockSize; i += blockSize) {
        int localX = i % macroBlockW;
        int localY = i / macroBlockW;

        int globalX = baseBlockX + localX;
        int globalY = baseBlockY + localY;

        const uchar4* rowPrev = (const uchar4*)((const char*)prevFrame + globalY * prevPitch);
        prevTile[i] = rowPrev[globalX];
    }

    range dx_range = {
        max(-baseBlockX, -search_halfsizeX),
        min(search_halfsizeX, width - (baseBlockX + macroBlockW))
    };

    range dy_range = {
        max(-baseBlockY, -search_halfsizeY),
        min(search_halfsizeY, height - (baseBlockY + macroBlockH))
    };

    int currTileW = macroBlockW + (dx_range.max - dx_range.min);
    int currTileH = macroBlockH + (dy_range.max - dy_range.min);

    int currTileSize = currTileW * currTileH;

    #pragma unroll 1
    for (int i = tid; i < currTileSize; i += blockSize) {
        int localX = i % currTileW;
        int localY = i / currTileW;

        int globalX = baseBlockX + localX + dx_range.min;
        int globalY = baseBlockY + localY + dy_range.min;

        const uchar4* rowCurr = (const uchar4*)((const char*)currFrame + globalY * currPitch);
        currTile[i] = rowCurr[globalX];
    }

    __syncthreads();

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;

    // collect full info now for testing
    SadStats localStats = EmptySadStats();

    int currTileBaseX = abs(dx_range.min);
    int currTileBaseY = abs(dy_range.min);

    for (int candidate = tid; candidate < allCandidates; candidate += blockSize) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        int sad = 0;

        //#pragma unroll -> makes it worse actually, too much unrolls
        #pragma unroll 1
        for (int i = 0; i < macroBlockH; i++) {
            uchar4* prev = prevTile + (i * macroBlockW);
            uchar4* curr = currTile + (i + currTileBaseY + dy) * currTileW;

            #pragma unroll
            for (int j = 0; j < macroBlockW; ++j) {
                int3 prevSample = make_int3(prev[j].x, prev[j].y, prev[j].z);

                const int currX = j + currTileBaseX + dx;
                int3 currSample = make_int3(curr[currX].x, curr[currX].y, curr[currX].z);

                int diffR = abs(prevSample.x - currSample.x);
                int diffG = abs(prevSample.y - currSample.y);
                int diffB = abs(prevSample.z - currSample.z);

                sad += (diffR + diffB + diffG);
            }
        }

        UpdateBestSad(localStats, { sad, dx, dy });
    }

    const int fullTileSize = (macroBlockW + search_halfsizeX * 2) * (macroBlockH + search_halfsizeY * 2);

    SadStats* sadTile = (SadStats*)(currTile + fullTileSize);

    bool valid = tid < allCandidates;
    sadTile[tid] = valid ? localStats : EmptySadStats();

    __syncthreads();

    #pragma unroll 1
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SadStats curr = sadTile[tid];
            SadStats next = sadTile[tid + stride];

            SadTop2 out = MergeSadTop2(curr.top2, next.top2);

            unsigned long sumSad = curr.sumSad + next.sumSad;
            int zeroSad = curr.zeroSad + next.zeroSad;

            sadTile[tid] = SadStats{ out, sumSad, zeroSad };
        }

        __syncthreads();
    }

    if (tid == 0) {
        BlockMatchStats* rowStats = (BlockMatchStats*)((char*)statsOut + blockIdx.y * statsOutPitch);

        BlockMatchStats out{
            { sadTile[0].top2.best.dx, sadTile[0].top2.best.dy },
            { sadTile[0].top2.second.dx, sadTile[0].top2.second.dy },

            sadTile[0].top2.best.sad,
            sadTile[0].top2.second.sad,

            sadTile[0].zeroSad,
            sadTile[0].sumSad
        };

        rowStats[blockIdx.x] = out;
    }
}

namespace detail {
    using image::GpuImageView;
    using cuda::motion::BlockMatchingParams;
    using cuda::motion::BlockMatchStats;

    void BlockMatching(
        const GpuImageView<uchar4>& prevFrame,
        const GpuImageView<uchar4>& currFrame,
        GpuImageView<BlockMatchStats>& statsOut,
        const BlockMatchingParams& p,
        cuda::KernelContext& ctx)
    {
        //static bool print = true;
        //if (print)
        //{
        //    PRINT_KERNEL_ATTRS(BlockMatchingGenericKernel);
        //    std::cout << std::endl;
        //    print = false;
        //}

        dim3 gridSize = cuda::math::Div(prevFrame.m_dim, p.macroBlockDim);

        const size_t prevTileSize = (p.macroBlockDim.x * p.macroBlockDim.y) * sizeof(uchar4);
        const size_t currTileSize = (p.macroBlockDim.x + p.search_halfsize.x * 2) * (p.macroBlockDim.y + p.search_halfsize.y * 2) * sizeof(uchar4);
        const size_t sadTileSize = (p.blockDim.x * p.blockDim.y) * sizeof(SadStats);

        const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
        image::vec2i macroBlockInt = { int(p.macroBlockDim.x), int(p.macroBlockDim.y) };

        cuda::TimedCall("BlockMatchingGenericKernel: " + cuda::util::BlockDimToString(p.blockDim), ctx, [&]() {
            BlockMatchingKernel <<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                prevFrame.m_ptr,
                prevFrame.m_pitch,
                currFrame.m_ptr,
                currFrame.m_pitch,
                statsOut.m_ptr,
                statsOut.m_pitch,
                prevFrame.m_dim.x,
                prevFrame.m_dim.y,
                p.search_halfsize,
                macroBlockInt
            );
        });
    }

    template <
        unsigned int blockDimX,
        unsigned int blockDimY,
        int macroBlockW,
        int macroBlockH,
        int search_halfsizeX,
        int search_halfsizeY
    >
    void BlockMatchingT(
        const GpuImageView<uchar4>& prevFrame,
        const GpuImageView<uchar4>& currFrame,
        GpuImageView<BlockMatchStats>& statsOut,
        cuda::KernelContext& ctx)
    {
        //static bool print = true;
        //if (print)
        //{
        //    PRINT_KERNEL_ATTRS((BlockMatchingKernelT<8, 8, 3, 3>));
        //    std::cout << std::endl;
        //    print = false;
        //}

        constexpr image::vec2ui macroBlockSize = { macroBlockW, macroBlockH };
        constexpr image::vec2ui cudaBlockDim = { blockDimX, blockDimY };
        constexpr image::vec2i search_halfsize = { search_halfsizeX, search_halfsizeY };

        dim3 gridSize = cuda::math::Div(prevFrame.m_dim, macroBlockSize);

        constexpr size_t prevTileSize = (macroBlockSize.x * macroBlockSize.y) * sizeof(uchar4);
        constexpr size_t currTileSize = (macroBlockSize.x + search_halfsize.x * 2) * (macroBlockSize.y + search_halfsize.y * 2) * sizeof(uchar4);
        constexpr size_t sadTileSize = (cudaBlockDim.x * cudaBlockDim.y) * sizeof(SadStats);

        constexpr size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;

        cuda::TimedCall("BlockMatchingKernelT: " + cuda::util::BlockDimToString(cudaBlockDim), ctx, [&]() {
            BlockMatchingKernelT<macroBlockW, macroBlockH, search_halfsizeX, search_halfsizeY> <<<gridSize, cuda::math::vec2Todim3(cudaBlockDim), sharedMemSize, ctx.m_stream>>> (
                prevFrame.m_ptr,
                prevFrame.m_pitch,
                currFrame.m_ptr,
                currFrame.m_pitch,
                statsOut.m_ptr,
                statsOut.m_pitch,
                prevFrame.m_dim.x,
                prevFrame.m_dim.y
            );
        });
    }
}

namespace cuda {
    namespace motion {
        template <class Vec>
        bool Eq(Vec vec1, Vec vec2) {
            return vec1.x == vec2.x && vec1.y == vec2.y;
        }

        bool EqParams(const BlockMatchingParams& p1, const BlockMatchingParams& p2) {
            return Eq(p1.blockDim, p2.blockDim)
                && Eq(p1.macroBlockDim, p2.macroBlockDim)
                && Eq(p1.search_halfsize, p2.search_halfsize);
        }

        void BlockMatching(
            const GpuImageView<uchar4>& prevFrame,
            const GpuImageView<uchar4>& currFrame,
            GpuImageView<BlockMatchStats>& statsOut,
            const BlockMatchingParams& p,
            cuda::KernelContext& ctx)
        {
            const std::vector<BlockMatchingParams> supportedTemplP {
                // blockDim; macroBlockDim; search_halfsize
                { { 8, 8 }, { 16, 16 }, { 3, 3 } },
                { { 8, 8 }, { 16, 16 }, { 4, 4 } },
                { { 8, 8 }, { 16, 16 }, { 5, 5 } },
                { { 8, 8 }, { 16, 16 }, { 6, 6 } },

                { { 16, 16 }, { 16, 16 }, { 3, 3 } },
            };

            if (EqParams(p, supportedTemplP[0])) {
                detail::BlockMatchingT<8, 8, 16, 16, 3, 3>(prevFrame, currFrame, statsOut, ctx);
            }
            else if (EqParams(p, supportedTemplP[1])) {
                detail::BlockMatchingT<8, 8, 16, 16, 4, 4>(prevFrame, currFrame, statsOut, ctx);
            }
            else if (EqParams(p, supportedTemplP[2])) {
                detail::BlockMatchingT<8, 8, 16, 16, 5, 5>(prevFrame, currFrame, statsOut, ctx);
            }
            else if (EqParams(p, supportedTemplP[3])) {
                detail::BlockMatchingT<8, 8, 16, 16, 6, 6>(prevFrame, currFrame, statsOut, ctx);
            }
            else if (EqParams(p, supportedTemplP[4])) {
                detail::BlockMatchingT<16, 16, 16, 16, 3, 3>(prevFrame, currFrame, statsOut, ctx);
            }
            else {
                detail::BlockMatching(prevFrame, currFrame, statsOut, p, ctx);
            }
        }
    }
}
