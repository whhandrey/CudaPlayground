#include "Motion.cuh"
#include "Common.cuh"
#include "../Common/TimedCudaCall.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace {
    struct SadVector {
        int sad;
        int dx;
        int dy;
        // TODO: maybe add padding 4 bytes?
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
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    int2* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height,
    int2 search_halfsize,
    int2 macroBlockDim)
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

        int2* rowMotion = (int2*)((char*)motionImg + blockIdx.y * motionPitch);
        rowMotion[blockIdx.x] = make_int2(finalMin.dx, finalMin.dy);
    }
}

__global__ void BlockMatchingSimpleKernel(
    const uchar4* __restrict__ prevFrame,
    size_t prevPitch,
    const uchar4* __restrict__ currFrame,
    size_t currPitch,
    int4* __restrict__ motionImg,
    size_t motionPitch,
    int width,
    int height,
    int2 search_halfsize,
    int2 macroBlockDim)
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

    SadVector bestSad = { INT_MAX, 0, 0 };

    int currTileBaseX = abs(dx_range.min);
    int currTileBaseY = abs(dy_range.min);

    for (int candidate = tid; candidate < allCandidates; candidate += blockSize) {
        int dx = (candidate % search_x) + dx_range.min;
        int dy = (candidate / search_x) + dy_range.min;

        int sad = 0;
        for (int i = 0; i < macroBlockDim.y; i++) {
            uchar4* prev = prevTile + (i * macroBlockDim.x);
            uchar4* curr = currTile + (i + currTileBaseY + dy) * currTileW;

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
        int4* rowMotion = (int4*)((char*)motionImg + blockIdx.y * motionPitch);

        // w is unused now
        rowMotion[blockIdx.x] = make_int4(sadTile[0].dx, sadTile[0].dy, sadTile[0].sad, 0);
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
    int4* __restrict__ motionImg,
    size_t motionPitch,
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
        int4* rowMotion = (int4*)((char*)motionImg + blockIdx.y * motionPitch);

        // w is unused now
        rowMotion[blockIdx.x] = make_int4(sadTile[0].dx, sadTile[0].dy, sadTile[0].sad, 0);
    }
}

__global__ void ShiftImageKernel(
    const uchar4* __restrict__ image,
    size_t imgPitch,
    uchar4* __restrict__ output,
    size_t outputPitch,
    int width,
    int height,
    int2 shiftVector)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int2 shifted = { x - shiftVector.x, y - shiftVector.y };
    bool valid = (shifted.x >= 0 && shifted.x < width) && (shifted.y >= 0 && shifted.y < height);

    uchar4 out_sample = {};

    if (valid) {
        const uchar4* inputRow = (const uchar4*)((const char*)image + shifted.y * imgPitch);
        out_sample = inputRow[shifted.x];
    }

    uchar4* outputRow = (uchar4*)((char*)output + y * outputPitch);
    outputRow[x] = out_sample;
}

namespace motion {
    namespace shift {
        ImageGPU<uchar4> ShiftImage(const ImageGPU<uchar4>& image, int2 shiftVector, cudaStream_t stream, ivec2 blockSize) {
            dim3 gridSize = Div(image.Dim(), blockSize);
            ImageGPU<uchar4> output(image.Dim());

            cuda::TimedCall("ShiftImageKernel: " + util::BlockDimToString(blockSize), stream, [&]() {
                ShiftImageKernel<<<gridSize, vec2Todim3(blockSize), 0, stream>>> (
                    image.Data(),
                    image.Pitch(),
                    output.Data(),
                    output.Pitch(),
                    image.Dim().x,
                    image.Dim().y,
                    shiftVector
                );
            });

            return output;
        }
    }

    ImageGPU<int4> BlockMatchingSimple(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 macroBlockSize, int2 search_halfsize, ivec2 cudaBlockDim, cudaStream_t stream) {
        dim3 gridSize = Div(prevFrame.Dim(), macroBlockSize);

        ImageGPU<int4> output({ gridSize.x, gridSize.y });

        const size_t prevTileSize = (macroBlockSize.x * macroBlockSize.y) * sizeof(uchar4);
        const size_t currTileSize = (macroBlockSize.x + search_halfsize.x * 2) * (macroBlockSize.y + search_halfsize.y * 2) * sizeof(uchar4);
        const size_t sadTileSize = (cudaBlockDim.x * cudaBlockDim.y) * sizeof(SadVector);

        const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
        ivec2 dim = prevFrame.Dim();

        int2 macroBlock = { int(macroBlockSize.x), int(macroBlockSize.y) };

        cuda::TimedCall("BlockMatchingSimpleKernel: " + util::BlockDimToString(cudaBlockDim), stream, [&]() {
            BlockMatchingSimpleKernel<<<gridSize, vec2Todim3(cudaBlockDim), sharedMemSize, stream>>> (
                prevFrame.Data(),
                prevFrame.Pitch(),
                currFrame.Data(),
                currFrame.Pitch(),
                output.Data(),
                output.Pitch(),
                dim.x,
                dim.y,
                search_halfsize,
                macroBlock
            );
        });

        return output;
    }

    ImageGPU<int4> BlockMatchingSimpleT(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 /*macroBlockSize*/, int2 /*search_halfsize*/, ivec2 /*cudaBlockDim*/, cudaStream_t stream) {
        constexpr int macroBlockW = 16;
        constexpr int macroBlockH = 16;

        constexpr int search_halfsizeX = 3;
        constexpr int search_halfsizeY = 3;

        constexpr ivec2 macroBlockSize = { macroBlockW, macroBlockH };
        constexpr ivec2 cudaBlockDim = { 8, 8 };
        constexpr int2 search_halfsize = { search_halfsizeX, search_halfsizeY };

        dim3 gridSize = Div(prevFrame.Dim(), macroBlockSize);

        ImageGPU<int4> output({ gridSize.x, gridSize.y });

        constexpr size_t prevTileSize = (macroBlockSize.x * macroBlockSize.y) * sizeof(uchar4);
        constexpr size_t currTileSize = (macroBlockSize.x + search_halfsize.x * 2) * (macroBlockSize.y + search_halfsize.y * 2) * sizeof(uchar4);
        constexpr size_t sadTileSize = (cudaBlockDim.x * cudaBlockDim.y) * sizeof(SadVector);

        constexpr size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
        ivec2 dim = prevFrame.Dim();

        cuda::TimedCall("BlockMatchingSimpleKernel_T: " + util::BlockDimToString(cudaBlockDim), stream, [&]() {
            BlockMatchingSimpleKernel_T<macroBlockW, macroBlockH, search_halfsizeX, search_halfsizeY> <<<gridSize, vec2Todim3(cudaBlockDim), sharedMemSize, stream>>> (
                prevFrame.Data(),
                prevFrame.Pitch(),
                currFrame.Data(),
                currFrame.Pitch(),
                output.Data(),
                output.Pitch(),
                dim.x,
                dim.y
            );
        });

        return output;
    }

	ImageGPU<int2> BlockMatchingWarp(const ImageGPU<uchar4>& prevFrame, const ImageGPU<uchar4>& currFrame, ivec2 macroBlockSize, int2 search_halfsize, ivec2 cudaBlockDim, cudaStream_t stream) {
        assert((cudaBlockDim.x * cudaBlockDim.y) % 32 == 0);
        
        dim3 gridSize = Div(prevFrame.Dim(), macroBlockSize);

        ImageGPU<int2> output({ gridSize.x, gridSize.y });

        const int warpSize = 32;
        const int numWarpsPerBlock = (cudaBlockDim.x * cudaBlockDim.y) / warpSize;

        const size_t prevTileSize = (macroBlockSize.x * macroBlockSize.y) * sizeof(uchar4);
        const size_t currTileSize = (macroBlockSize.x + search_halfsize.x * 2) * (macroBlockSize.y + search_halfsize.y * 2) * sizeof(uchar4);
        const size_t sadTileSize = numWarpsPerBlock * sizeof(SadVector);

        const size_t sharedMemSize = prevTileSize + currTileSize + sadTileSize;
        ivec2 dim = prevFrame.Dim();

        int2 macroBlock = { int(macroBlockSize.x), int(macroBlockSize.y) };

        cuda::TimedCall("BlockMatchingKernel: " + util::BlockDimToString(cudaBlockDim), stream, [&]() {
            BlockMatchingWarpKernel<<<gridSize, vec2Todim3(cudaBlockDim), sharedMemSize, stream>>> (
                prevFrame.Data(),
                prevFrame.Pitch(),
                currFrame.Data(),
                currFrame.Pitch(),
                output.Data(),
                output.Pitch(),
                dim.x,
                dim.y,
                search_halfsize,
                macroBlock
            );
        });

        return output;
	}
}
