#include "Motion.h"
#include "Common.cuh"
#include "KernelCommon.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

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
    struct SadCandidate {
        int sad;
        int dx;
        int dy;
    };

    struct SadTop2 {
        SadCandidate best;
        SadCandidate second;
    };

    struct SadStats {
        SadTop2 top2;
        unsigned long long sum;
    };

    struct ALIGN(8) range {
        int min;
        int max;
    };
}

__device__ __forceinline__ bool ValidCandidate(SadCandidate c) {
    return c.sad != INT_MAX;
}

__device__ __forceinline__ SadCandidate MinSad(SadCandidate first, SadCandidate second) {
    return first.sad < second.sad ? first : second;
}

__device__ __forceinline__ SadCandidate EmptySadCand() {
    return { INT_MAX, 0, 0 };
}

__device__ __forceinline__ SadTop2 EmptySadTop2() {
    return { EmptySadCand(), EmptySadCand() };
}

__device__ __forceinline__ SadStats EmptySadStats() {
    return {
        EmptySadTop2(),
        0
    };
}

__device__ __forceinline__ void UpdateBestSad(SadStats& sadLocal, SadCandidate sadCand) {
    if (sadCand.sad < sadLocal.top2.best.sad) {
        sadLocal.top2.second = sadLocal.top2.best;
        sadLocal.top2.best = sadCand;
    }
    else if (sadCand.sad < sadLocal.top2.second.sad) {
        sadLocal.top2.second = sadCand;
    }

    sadLocal.sum += sadCand.sad;
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

__global__  void BlockMatchingSimpleKernel(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    unsigned char* __restrict__ conf,
    size_t confPitch,
    int2* __restrict__ dxdyOutput,
    size_t dxdyOutPitch,
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

                // not a huge diff how you write this crap.
                // does not matter. for blackwell and compute_120 now floats mean smth
                // 
                //const uchar4 p = prev[j];
                //const uchar4 c = curr[currX];

                //const int diffR = abs(int(p.x) - int(c.x));
                //const int diffG = abs(int(p.y) - int(c.y));
                //const int diffB = abs(int(p.z) - int(c.z));

                //sad += diffR + diffG + diffB;
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
            unsigned long long sum = curr.sum + next.sum;

            sadTile[tid] = SadStats{ out, sum };
        }

        __syncthreads();
    }

    if (tid == 0) {
        if (dxdyOutput) {
            int2* rowDxDyVec = (int2*)((char*)dxdyOutput + blockIdx.y * dxdyOutPitch);
            rowDxDyVec[blockIdx.x] = make_int2(sadTile[0].top2.best.dx, sadTile[0].top2.best.dy);
        }

        unsigned char* rowConf = (unsigned char*)((char*)conf + blockIdx.y * confPitch);

        float avg_sad = sadTile[0].sum / float(allCandidates);
        float min_sad = float(sadTile[0].top2.best.sad);

        float conf = 0.0f;
        if (avg_sad > 1e-10f) {
            conf = 1.0f - min_sad / avg_sad;

        }

        rowConf[blockIdx.x] = unsigned char(saturate(conf) * 255.0f + 0.5f);
    }

}

template <
    int macroBlockW,
    int macroBlockH,
    int search_halfsizeX,
    int search_halfsizeY
>
__global__ void BlockMatchingSimpleKernel_T(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    unsigned char* __restrict__ conf,
    size_t confPitch,
    int2* __restrict__ dxdyOutput,
    size_t dxdyOutPitch,
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

    constexpr int fullTileSize = (macroBlockW + search_halfsizeX * 2) * (macroBlockH + search_halfsizeY * 2);

    SadStats* sadTile = (SadStats*)(currTile + fullTileSize);

    bool valid = tid < allCandidates;
    sadTile[tid] = valid ? localStats : EmptySadStats();

    __syncthreads();

    // this is still faster than reducing 64 values (for 8x8 cuda block) in one thread
    #pragma unroll 1
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SadStats curr = sadTile[tid];
            SadStats next = sadTile[tid + stride];

            SadTop2 out = MergeSadTop2(curr.top2, next.top2);
            unsigned long long sum = curr.sum + next.sum;

            sadTile[tid] = SadStats{ out, sum };
        }

        __syncthreads();
    }

    if (tid == 0) {
        if (dxdyOutput) {
            int2* rowDxDyVec = (int2*)((char*)dxdyOutput + blockIdx.y * dxdyOutPitch);
            rowDxDyVec[blockIdx.x] = make_int2(sadTile[0].top2.best.dx, sadTile[0].top2.best.dy);
        }

        unsigned char* rowConf = (unsigned char*)((char*)conf + blockIdx.y * confPitch);

        float avg_sad = sadTile[0].sum / float(allCandidates);
        float min_sad = float(sadTile[0].top2.best.sad);

        float conf = 0.0f;
        if (avg_sad > 1e-10f) {
            conf = 1.0f - min_sad / avg_sad;

        }
        
        rowConf[blockIdx.x] = unsigned char(saturate(conf) * 255.0f + 0.5f);
    }
}

__global__ void ShiftImageKernel(
    const uchar4* __restrict__ image,
    size_t imgPitch,
    uchar4* __restrict__ output,
    size_t outputPitch,
    int width,
    int height,
    image::vec2i shiftVector)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    image::vec2i shifted = { x - shiftVector.x, y - shiftVector.y };
    bool valid = (shifted.x >= 0 && shifted.x < width) && (shifted.y >= 0 && shifted.y < height);

    uchar4 out_sample = {};

    if (valid) {
        const uchar4* inputRow = (const uchar4*)((const char*)image + shifted.y * imgPitch);
        out_sample = inputRow[shifted.x];
    }

    uchar4* outputRow = (uchar4*)((char*)output + y * outputPitch);
    outputRow[x] = out_sample;
}

namespace cuda {
    namespace motion {
        namespace shift {
            void ShiftImage(const GpuImageView<uchar4>& image, GpuImageView<uchar4>& output, image::vec2i shiftVector, cuda::KernelContext& ctx, image::vec2ui blockSize) {
                dim3 gridSize = cuda::math::Div(image.m_dim, blockSize);

                cuda::TimedCall("ShiftImageKernel: " + cuda::util::BlockDimToString(blockSize), ctx, [&]() {
                    ShiftImageKernel<<<gridSize, cuda::math::vec2Todim3(blockSize), 0, ctx.m_stream>>> (
                        image.m_ptr,
                        image.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        image.m_dim.x,
                        image.m_dim.y,
                        shiftVector
                    );
                });
            }
        }

        void BlockMatchingSimple(
            const GpuImageView<uchar4>& prevFrame,
            const GpuImageView<uchar4>& currFrame,
            GpuImageView<unsigned char>& confOut,
            GpuImageView<int2>& dxdyOut,
            const BlockMatchingParams& p,
            cuda::KernelContext& ctx)
        {
            //static bool print = true;
            //if (print)
            //{
            //    PRINT_KERNEL_ATTRS(BlockMatchingSimpleKernel);
            //    std::cout << std::endl;
            //    print = false;
            //}

            dim3 gridSize = cuda::math::Div(prevFrame.m_dim, p.macroBlockDim);

            const size_t prevTileSize = (p.macroBlockDim.x * p.macroBlockDim.y) * sizeof(uchar4);
            const size_t currTileSize = (p.macroBlockDim.x + p.search_halfsize.x * 2) * (p.macroBlockDim.y + p.search_halfsize.y * 2) * sizeof(uchar4);
            const size_t sadTileSize = (p.blockDim.x * p.blockDim.y) * sizeof(SadStats);

            const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
            image::vec2i macroBlockInt = { int(p.macroBlockDim.x), int(p.macroBlockDim.y) };

            cuda::TimedCall("BlockMatchingSimpleKernel: " + cuda::util::BlockDimToString(p.blockDim), ctx, [&]() {
                BlockMatchingSimpleKernel<<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                    prevFrame.m_ptr,
                    prevFrame.m_pitch,
                    currFrame.m_ptr,
                    currFrame.m_pitch,
                    confOut.m_ptr,
                    confOut.m_pitch,
                    dxdyOut.m_ptr,
                    dxdyOut.m_pitch,
                    prevFrame.m_dim.x,
                    prevFrame.m_dim.y,
                    p.search_halfsize,
                    macroBlockInt
                );
            });
        }

        void BlockMatchingSimpleT(
            const GpuImageView<uchar4>& prevFrame,
            const GpuImageView<uchar4>& currFrame,
            GpuImageView<unsigned char>& confOut,
            GpuImageView<int2>& dxdyOut,
            const BlockMatchingParams& /*p*/,
            cuda::KernelContext& ctx)
        {
            //static bool print = true;
            //if (print)
            //{
            //    PRINT_KERNEL_ATTRS((BlockMatchingSimpleKernel_T<8, 8, 3, 3>));
            //    std::cout << std::endl;
            //    print = false;
            //}

            constexpr int macroBlockW = 16;
            constexpr int macroBlockH = 16;

            constexpr int search_halfsizeX = 3;
            constexpr int search_halfsizeY = 3;

            constexpr image::vec2ui macroBlockSize = { macroBlockW, macroBlockH };
            constexpr image::vec2ui cudaBlockDim = { 8, 8 };
            constexpr image::vec2i search_halfsize = { search_halfsizeX, search_halfsizeY };

            dim3 gridSize = cuda::math::Div(prevFrame.m_dim, macroBlockSize);

            constexpr size_t prevTileSize = (macroBlockSize.x * macroBlockSize.y) * sizeof(uchar4);
            constexpr size_t currTileSize = (macroBlockSize.x + search_halfsize.x * 2) * (macroBlockSize.y + search_halfsize.y * 2) * sizeof(uchar4);
            constexpr size_t sadTileSize = (cudaBlockDim.x * cudaBlockDim.y) * sizeof(SadStats);

            constexpr size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;

            cuda::TimedCall("BlockMatchingSimpleKernel_T: " + cuda::util::BlockDimToString(cudaBlockDim), ctx, [&]() {
                BlockMatchingSimpleKernel_T<macroBlockW, macroBlockH, search_halfsizeX, search_halfsizeY> <<<gridSize, cuda::math::vec2Todim3(cudaBlockDim), sharedMemSize, ctx.m_stream>>> (
                    prevFrame.m_ptr,
                    prevFrame.m_pitch,
                    currFrame.m_ptr,
                    currFrame.m_pitch,
                    confOut.m_ptr,
                    confOut.m_pitch,
                    dxdyOut.m_ptr,
                    dxdyOut.m_pitch,
                    prevFrame.m_dim.x,
                    prevFrame.m_dim.y
                );
            });
        }

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

            cuda::TimedCall("BlockMatchingWarpKernel: " + cuda::util::BlockDimToString(p.blockDim), ctx, [&]() {
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
