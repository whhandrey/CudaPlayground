#include "Motion.h"
#include "Common.cuh"
#include "KernelCommon.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

namespace {
    struct SadVector {
        int sad;
        int dx;
        int dy;
        // TODO: maybe add padding 4 bytes?
    };

    struct ALIGN(8) range {
        int min;
        int max;
    };
}

__device__ __forceinline__ SadVector MinSad(SadVector first, SadVector second) {
    return first.sad < second.sad ? first : second;
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

__global__ void BlockMatchingWarpKernel(
    const image::vec4uc* __restrict__ prevFrame,
    size_t prevPitch,
    const image::vec4uc* __restrict__ currFrame,
    size_t currPitch,
    image::vec2i* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height,
    image::vec2i search_halfsize,
    image::vec2i macroBlockDim)
{
    extern __shared__ image::vec4uc tiles[];

    image::vec4uc* prevTile = tiles;
    image::vec4uc* currTile = tiles + (macroBlockDim.x * macroBlockDim.y);

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

        const image::vec4uc* rowPrev = (const image::vec4uc*)((const char*)prevFrame + globalY * prevPitch);
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

        const image::vec4uc* rowCurr = (const image::vec4uc*)((const char*)currFrame + globalY * currPitch);
        currTile[i] = rowCurr[globalX];
    }

    __syncthreads();

    const int laneId = tid & 31; // tid % 32
    const int warpId = tid >> 5; // tid / 32
    const int warpsPerBlock = blockSize >> 5; // blockSize / 32

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;
    SadVector bestSad = { INT_MAX, 0, 0 };

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
    SadVector* sadTile = (SadVector*)(currTile + fullTileSize);

    if (laneId == 0) {
        sadTile[warpId] = bestSad;
    }

    __syncthreads();

    if (tid == 0) {
        SadVector finalMin = { INT_MAX, 0, 0 };

        for (int i = 0; i < warpsPerBlock; ++i) {
            finalMin = MinSad(finalMin, sadTile[i]);
        }

        image::vec2i* rowMotion = (image::vec2i*)((char*)motionImg + blockIdx.y * motionPitch);
        rowMotion[blockIdx.x] = { finalMin.dx, finalMin.dy };
    }
}

__global__ void BlockMatchingSimpleKernel(
    const image::vec4uc* __restrict__ prevFrame,
    size_t prevPitch,
    const image::vec4uc* __restrict__ currFrame,
    size_t currPitch,
    image::vec4i* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height,
    image::vec2i search_halfsize,
    image::vec2i macroBlockDim)
{
    extern __shared__ image::vec4uc tiles[];

    image::vec4uc* prevTile = tiles;
    image::vec4uc* currTile = prevTile + (macroBlockDim.x * macroBlockDim.y);

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

        const image::vec4uc* rowPrev = (const image::vec4uc*)((const char*)prevFrame + globalY * prevPitch);
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

        const image::vec4uc* rowCurr = (const image::vec4uc*)((const char*)currFrame + globalY * currPitch);
        currTile[i] = rowCurr[globalX];
    }

    __syncthreads();

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;

    SadVector bestSad = { INT_MAX, 0, 0 };

    int currTileBaseX = abs(dx_range.min);
    int currTileBaseY = abs(dy_range.min);

    for (int candidate = tid; candidate < allCandidates; candidate += blockSize) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        int sad = 0;
        for (int i = 0; i < macroBlockDim.y; i++) {
            image::vec4uc* prev = prevTile + (i * macroBlockDim.x);
            image::vec4uc* curr = currTile + (i + currTileBaseY + dy) * currTileW;

            for (int j = 0; j < macroBlockDim.x; ++j) {
                int3 prevSample = make_int3(prev[j].x, prev[j].y, prev[j].z);

                const int currX = j + currTileBaseX + dx;
                int3 currSample = make_int3(curr[currX].x, curr[currX].y, curr[currX].z);

                int diffR = abs(prevSample.x - currSample.x);
                int diffG = abs(prevSample.y - currSample.y);
                int diffB = abs(prevSample.z - currSample.z);

                sad += (diffR + diffB + diffG);
            }
        }

        if (sad < bestSad.sad) {
            bestSad = { sad, dx, dy };
        }
    }

    int fullTileSize = (macroBlockDim.x + search_halfsize.x * 2) * (macroBlockDim.y + search_halfsize.y * 2);

    SadVector* sadTile = (SadVector*)(currTile + fullTileSize);

    bool valid = tid < allCandidates;
    sadTile[tid] = valid ? bestSad : SadVector{ INT_MAX, 0, 0 };

    __syncthreads();

    // this is still faster than reducing 64 values (for 8x8 cuda block) in one thread
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SadVector curr = sadTile[tid];
            SadVector next = sadTile[tid + stride];

            sadTile[tid] = MinSad(curr, next);
        }

        __syncthreads();
    }

    if (tid == 0) {
        image::vec4i* rowMotion = (image::vec4i*)((char*)motionImg + blockIdx.y * motionPitch);

        // w is unused now
        rowMotion[blockIdx.x] = { sadTile[0].dx, sadTile[0].dy, sadTile[0].sad, 0 };
    }
}

template <
    int macroBlockW,
    int macroBlockH,
    int search_halfsizeX,
    int search_halfsizeY
>
__global__ void BlockMatchingSimpleKernel_T(
    const image::vec4uc* __restrict__ prevFrame,
    size_t prevPitch,
    const image::vec4uc* __restrict__ currFrame,
    size_t currPitch,
    image::vec4i* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height)
{
    extern __shared__ image::vec4uc tiles[];

    image::vec4uc* prevTile = tiles;
    image::vec4uc* currTile = prevTile + (macroBlockW * macroBlockH);

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

        const image::vec4uc* rowPrev = (const image::vec4uc*)((const char*)prevFrame + globalY * prevPitch);
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

        const image::vec4uc* rowCurr = (const image::vec4uc
            *)((const char*)currFrame + globalY * currPitch);
        currTile[i] = rowCurr[globalX];
    }

    __syncthreads();

    int search_x = (dx_range.max - dx_range.min) + 1;
    int search_y = (dy_range.max - dy_range.min) + 1;

    const int allCandidates = search_x * search_y;

    SadVector bestSad = { INT_MAX, 0, 0 };

    int currTileBaseX = abs(dx_range.min);
    int currTileBaseY = abs(dy_range.min);

    for (int candidate = tid; candidate < allCandidates; candidate += blockSize) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        int sad = 0;

        //#pragma unroll -> makes it worse actually, too much unrolls
        #pragma unroll 1
        for (int i = 0; i < macroBlockH; i++) {
            image::vec4uc* prev = prevTile + (i * macroBlockW);
            image::vec4uc* curr = currTile + (i + currTileBaseY + dy) * currTileW;

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

        if (sad < bestSad.sad) {
            bestSad = { sad, dx, dy };
        }
    }

    constexpr int fullTileSize = (macroBlockW + search_halfsizeX * 2) * (macroBlockH + search_halfsizeY * 2);

    SadVector* sadTile = (SadVector*)(currTile + fullTileSize);

    bool valid = tid < allCandidates;
    sadTile[tid] = valid ? bestSad : SadVector{ INT_MAX, 0, 0 };

    __syncthreads();

    // this is still faster than reducing 64 values (for 8x8 cuda block) in one thread
    #pragma unroll 1
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SadVector curr = sadTile[tid];
            SadVector next = sadTile[tid + stride];

            sadTile[tid] = MinSad(curr, next);
        }

        __syncthreads();
    }

    if (tid == 0) {
        image::vec4i* rowMotion = (image::vec4i*)((char*)motionImg + blockIdx.y * motionPitch);

        // w is unused now
        rowMotion[blockIdx.x] = { sadTile[0].dx, sadTile[0].dy, sadTile[0].sad, 0 };
    }
}

__global__ void ShiftImageKernel(
    const image::vec4uc* __restrict__ image,
    size_t imgPitch,
    image::vec4uc* __restrict__ output,
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

    image::vec4uc out_sample = {};

    if (valid) {
        const image::vec4uc* inputRow = (const image::vec4uc*)((const char*)image + shifted.y * imgPitch);
        out_sample = inputRow[shifted.x];
    }

    image::vec4uc* outputRow = (image::vec4uc*)((char*)output + y * outputPitch);
    outputRow[x] = out_sample;
}

namespace cuda {
    namespace motion {
        namespace shift {
            void ShiftImage(const GpuImageView<image::vec4uc>& image, GpuImageView<image::vec4uc>& output, image::vec2i shiftVector, cuda::KernelContext& ctx, image::vec2ui blockSize) {
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
            const GpuImageView<image::vec4uc>& prevFrame,
            const GpuImageView<image::vec4uc>& currFrame,
            GpuImageView<image::vec4i>& output,
            const BlockMatchingParams& p,
            cuda::KernelContext& ctx)
        {
            dim3 gridSize = cuda::math::Div(prevFrame.m_dim, p.macroBlockDim);

            const size_t prevTileSize = (p.macroBlockDim.x * p.macroBlockDim.y) * sizeof(uchar4);
            const size_t currTileSize = (p.macroBlockDim.x + p.search_halfsize.x * 2) * (p.macroBlockDim.y + p.search_halfsize.y * 2) * sizeof(uchar4);
            const size_t sadTileSize = (p.blockDim.x * p.blockDim.y) * sizeof(SadVector);

            const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
            image::vec2i macroBlockInt = { int(p.macroBlockDim.x), int(p.macroBlockDim.y) };

            cuda::TimedCall("BlockMatchingSimpleKernel: " + cuda::util::BlockDimToString(p.blockDim), ctx, [&]() {
                BlockMatchingSimpleKernel<<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
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

        void BlockMatchingSimpleT(
            const GpuImageView<image::vec4uc>& prevFrame,
            const GpuImageView<image::vec4uc>& currFrame,
            GpuImageView<image::vec4i>& output,
            const BlockMatchingParams& /*p*/,
            cuda::KernelContext& ctx)
        {
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
            constexpr size_t sadTileSize = (cudaBlockDim.x * cudaBlockDim.y) * sizeof(SadVector);

            constexpr size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;

            cuda::TimedCall("BlockMatchingSimpleKernel_T: " + cuda::util::BlockDimToString(cudaBlockDim), ctx, [&]() {
                BlockMatchingSimpleKernel_T<macroBlockW, macroBlockH, search_halfsizeX, search_halfsizeY> <<<gridSize, cuda::math::vec2Todim3(cudaBlockDim), sharedMemSize, ctx.m_stream>>> (
                    prevFrame.m_ptr,
                    prevFrame.m_pitch,
                    currFrame.m_ptr,
                    currFrame.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    prevFrame.m_dim.x,
                    prevFrame.m_dim.y
                );
            });
        }

        void BlockMatchingWarp(
            const GpuImageView<image::vec4uc>& prevFrame,
            const GpuImageView<image::vec4uc>& currFrame,
            GpuImageView<image::vec2i>& output,
            const BlockMatchingParams& p,
            cuda::KernelContext& ctx)
        {
            assert((cudaBlockDim.x * cudaBlockDim.y) % 32 == 0);

            dim3 gridSize = cuda::math::Div(prevFrame.m_dim, p.macroBlockDim);

            const int warpSize = 32;
            const int numWarpsPerBlock = (p.blockDim.x * p.blockDim.y) / warpSize;

            const size_t prevTileSize = (p.macroBlockDim.x * p.macroBlockDim.y) * sizeof(uchar4);
            const size_t currTileSize = (p.macroBlockDim.x + p.search_halfsize.x * 2) * (p.macroBlockDim.y + p.search_halfsize.y * 2) * sizeof(uchar4);
            const size_t sadTileSize = numWarpsPerBlock * sizeof(SadVector);

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
