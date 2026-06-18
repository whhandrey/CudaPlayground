#include "Filter.cuh"
#include "Common.cuh"
#include "../Common/TimedCudaCall.h"
#include <iostream>
#include <algorithm>
#include <numeric>

constexpr int MAX_FILTER_HALFSIZE = 10;
constexpr int MAX_KERNEL_SIZE = 2 * MAX_FILTER_HALFSIZE + 1;

__constant__ float c_spatial2D[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// TODO: this is better to clean up/move
#define MAX_GAUSS_WEIGHT_SIZE 20
__constant__ float c_weightsGauss[MAX_GAUSS_WEIGHT_SIZE];

#define COLOR_DIST_TABLE_SIZE 256
// Not good stuff cause constant memory may suck for divergent range LUT
//__constant__ float c_colorRangeLUT[COLOR_DIST_TABLE_SIZE];

namespace gauss {
    std::vector<float> MakeGaussWeights(int half, float sigma) {
        const int size = half + 1;
        std::vector<float> weights(size);

        for (int i = 0; i < size; ++i) {
            float x = static_cast<float>(i);
            float w = std::exp(-(x * x) / (2.0f * sigma * sigma));

            weights[i] = w;
        }

        return weights;
    }

    std::vector<float> MakeNormGaussWeights(int half, float sigma) {
        auto weights = MakeGaussWeights(half, sigma);
        const float sum = 2.0f * std::accumulate(weights.begin() + 1, weights.end(), 0.0f) + weights[0];

        std::transform(weights.begin(), weights.end(), weights.begin(), [sum](const float weight) {
            return weight / sum;
        });

        return weights;
    }

    std::vector<float> Make2dGaussWeights(int half, float sigma) {
        const int size1d = half * 2 + 1;
        std::vector<float> spatialWeights(size1d * size1d);

        for (int ky = 0; ky < size1d; ++ky) {
            int dy = ky - half;

            for (int kx = 0; kx < size1d; ++kx) {
                int dx = kx - half;

                float dist = float(dx * dx + dy * dy);
                float w = std::exp(-dist / (2.0f * sigma * sigma));

                spatialWeights[kx + ky * size1d] = w;
            }
        }

        return spatialWeights;
    }
}

namespace bilateral {
    // Assuming RGB
    std::vector<float> RangeLUT(float sigmaColor) {
        const int numColors = COLOR_DIST_TABLE_SIZE;
        std::vector<float> output(numColors);

        for (int i = 0; i < numColors; ++i) {
            float x = static_cast<float>(i);
            output[i] = std::exp(-(x * x) / (2.0f * sigmaColor * sigmaColor));
        }

        return output;
    }
}

__global__ void GaussianBlurX(
    const uchar4* src,
    size_t srcPitch,
    float4* dst,
    size_t dstPitch,
    int width,
    int height,
    int filter_halfsize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

    const uchar4* srcRow = (const uchar4*)((const char*)src + y * srcPitch);
    for (int dx = -filter_halfsize; dx <= filter_halfsize; ++dx) {
        int idx = clamp(x + dx, 0, width - 1);

        uchar4 p = srcRow[idx];
        float4 sample = make_float4(float(p.x), float(p.y), float(p.z), 0.0f);

        sum.x += (sample.x * c_weightsGauss[abs(dx)]);
        sum.y += (sample.y * c_weightsGauss[abs(dx)]);
        sum.z += (sample.z * c_weightsGauss[abs(dx)]);
    }

    float4* dstRow = (float4*)((char*)dst + y * dstPitch);
    dstRow[x] = make_float4(sum.x, sum.y, sum.z, float(srcRow[x].w));
}

__global__ void GaussianBlurY(
    const float4* src,
    size_t srcPitch,
    uchar4* dst,
    size_t dstPitch,
    int width,
    int height,
    int filter_halfsize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int dy = -filter_halfsize; dy <= filter_halfsize; ++dy) {
        int colIdx = clamp(y + dy, 0, height - 1);
        const float4* srcRow = (const float4*)((const char*)src + colIdx * srcPitch);

        float4 sample = srcRow[x];

        sum.x += (sample.x * c_weightsGauss[abs(dy)]);
        sum.y += (sample.y * c_weightsGauss[abs(dy)]);
        sum.z += (sample.z * c_weightsGauss[abs(dy)]);
    }

    const float4* srcRow = (const float4*)((const char*)src + y * srcPitch);
    uchar4* dstRow = (uchar4*)((char*)dst + y * dstPitch);

    uchar4 output = float4ToUchar4(sum);
    output.w = unsigned char(srcRow[x].w + 0.5f);

    dstRow[x] = output;
}

__global__ void GaussianBlurTile(
    const uchar4* src,
    size_t srcPitch,
    uchar4* dst,
    size_t dstPitch,
    int width,
    int height,
    int filter_halfsize)
{
    extern __shared__ float4 tile[];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // cause you want to skip halo + you need to shift threadIdx from halo
    int gy = blockIdx.y * (blockDim.y - 2 * filter_halfsize) + threadIdx.y - filter_halfsize;
    int loadY = clamp(gy, 0, height - 1);

    // loadY is clamped no need to check
    if (gx < width) {
        float4 sum = {};
        const uchar4* srcRow = (const uchar4*)((const char*)src + loadY * srcPitch);

        for (int dx = -filter_halfsize; dx <= filter_halfsize; ++dx) {
            int x_idx{ clamp(gx + dx, 0, width - 1) };

            uchar4 p = srcRow[x_idx];
            float4 sample = make_float4(float(p.x), float(p.y), float(p.z), 0.0f);

            sum.x += (sample.x * c_weightsGauss[abs(dx)]);
            sum.y += (sample.y * c_weightsGauss[abs(dx)]);
            sum.z += (sample.z * c_weightsGauss[abs(dx)]);
        }

        sum.w = float(srcRow[gx].w);
        tile[threadIdx.x + threadIdx.y * blockDim.x] = sum;
    }

    __syncthreads();

    // vertical pass excluding halo in y-dir
    bool inCenter = threadIdx.y >= filter_halfsize && threadIdx.y < blockDim.y - filter_halfsize;

    // no need to check that gy >= 0 cause gy == 0 starting when threadIdx.y == filter_halfsize.
    // and here we deliberately filter only center samples excl. halo.
    // gy < height has to be checked cause global coords should be in sync with local coords.
    if (inCenter && gx < width && gy < height) {
        float4 sum = {};

        for (int dy = -filter_halfsize; dy <= filter_halfsize; ++dy) {
            int flat_index = threadIdx.x + (threadIdx.y + dy) * blockDim.x;

            float4 sample = tile[flat_index];

            sum.x += (sample.x * c_weightsGauss[abs(dy)]);
            sum.y += (sample.y * c_weightsGauss[abs(dy)]);
            sum.z += (sample.z * c_weightsGauss[abs(dy)]);
        }

        int flat_index = threadIdx.x + threadIdx.y * blockDim.x;

        uchar4 result = float4ToUchar4(sum);
        result.w = tile[flat_index].w;

        uchar4* dstRow = (uchar4*)((char*)dst + gy * dstPitch);
        dstRow[gx] = result;
    }
}

__global__ void BilateralKernel(
    const uchar4* __restrict__ src,
    size_t srcPitch,
    uchar4* __restrict__ dst,
    size_t dstPitch,
    int width,
    int height,
    int filter_halfsize,
    const float* __restrict__ colorRangeLUT)
{
    extern __shared__ int4 bilateralTile[];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    int tileWidth = blockDim.x + 2 * filter_halfsize;
    int tileHeight = blockDim.y + 2 * filter_halfsize;

    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;

    // last right-bottom block(s) is fully load but not fully used.
    // only if img.dim is fully divisible by blockDim then fully used.
    for (int i = threadId; i < tileWidth * tileHeight; i += blockSize) {
        int tileX = i % tileWidth;
        int tileY = i / tileWidth;

        int srcX = clamp(int(blockIdx.x * blockDim.x) + tileX - filter_halfsize, 0, width - 1);
        int srcY = clamp(int(blockIdx.y * blockDim.y) + tileY - filter_halfsize, 0, height - 1);

        const uchar4* row = (const uchar4*)((const char*)src + srcY * srcPitch);
        uchar4 p = __ldg(&row[srcX]);

        int lumaPart = (77 * p.x + 150 * p.y + 29 * p.z) >> 8;
        bilateralTile[i] = make_int4(p.x, p.y, p.z, lumaPart);
    }

    __syncthreads();

    if (gx >= width || gy >= height) {
        return;
    }

    float4 sum = {};
    float weightSum = 0.0f;

    int tileCenterX = threadIdx.x + filter_halfsize;
    int tileCenterY = threadIdx.y + filter_halfsize;

    //const int size1d = filter_halfsize * 2 + 1;
    int4 center = bilateralTile[tileCenterX + tileCenterY * tileWidth];

    const int K = filter_halfsize * 2 + 1;
    for (int ky = 0; ky < K; ++ky) {
        int rowBase = (threadIdx.y + ky) * tileWidth + threadIdx.x;
        int spatialBase = ky * K;

        for (int kx = 0; kx < K; ++kx) {
            int4 sample = bilateralTile[rowBase + kx];

            float spatialWeight = c_spatial2D[spatialBase + kx];

            int dY = abs(center.w - sample.w);
            float colorWeight = __ldg(&colorRangeLUT[dY]);

            float finalWeight = spatialWeight * colorWeight;

            sum.x += float(sample.x) * finalWeight;
            sum.y += float(sample.y) * finalWeight;
            sum.z += float(sample.z) * finalWeight;

            weightSum += finalWeight;
        }
    }

    float invWeight = 1.0f / weightSum;
    sum = { sum.x * invWeight, sum.y * invWeight, sum.z * invWeight };

    const uchar4* row = (const uchar4*)((const char*)src + gy * srcPitch);

    uchar4 out = float4ToUchar4(sum);
    out.w = row[gx].w;

    uchar4* dstRow = (uchar4*)((char*)dst + gy * dstPitch);
    dstRow[gx] = out;
}

template <int R>
__global__ void BilateralKernelT(
    const uchar4* __restrict__ src,
    size_t srcPitch,
    uchar4* __restrict__ dst,
    size_t dstPitch,
    int width,
    int height,
    const float* __restrict__ colorRangeLUT)
{
    extern __shared__ int4 bilateralTile[];

    const int tileWidth = blockDim.x + 2 * R;
    const int tileHeight = blockDim.y + 2 * R;
    const int blockSize = blockDim.x * blockDim.y;

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    const int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    // last right-bottom block(s) is fully load but not fully used.
    // only if img.dim is fully divisible by blockDim then fully used.
    for (int i = threadId; i < tileWidth * tileHeight; i += blockSize) {
        int tileX = i % tileWidth;
        int tileY = i / tileWidth;

        int srcX = clamp(int(blockIdx.x * blockDim.x) + tileX - R, 0, width - 1);
        int srcY = clamp(int(blockIdx.y * blockDim.y) + tileY - R, 0, height - 1);

        const uchar4* row = (const uchar4*)((const char*)src + srcY * srcPitch);
        uchar4 p = __ldg(&row[srcX]);

        int lumaPart = (77 * p.x + 150 * p.y + 29 * p.z) >> 8;
        bilateralTile[i] = { p.x, p.y, p.z, lumaPart };
    }

    __syncthreads();

    if (gx >= width || gy >= height) {
        return;
    }

    float4 sum = {};
    float weightSum = 0.0f;

    int tileCenterX = threadIdx.x + R;
    int tileCenterY = threadIdx.y + R;

    int4 center = bilateralTile[tileCenterX + tileCenterY * tileWidth];
    constexpr int K = 2 * R + 1;

    #pragma unroll
    for (int ky = 0; ky < K; ++ky) {
        int rowBase = (threadIdx.y + ky) * tileWidth + threadIdx.x;
        int spatialBase = ky * K;

        #pragma unroll
        for (int kx = 0; kx < K; ++kx) {
            int4 sample = bilateralTile[rowBase + kx];

            float spatialWeight = c_spatial2D[spatialBase + kx];

            int dY = abs(center.w - sample.w);
            float colorWeight = __ldg(&colorRangeLUT[dY]);

            float finalWeight = spatialWeight * colorWeight;

            sum.x += float(sample.x) * finalWeight;
            sum.y += float(sample.y) * finalWeight;
            sum.z += float(sample.z) * finalWeight;

            weightSum += finalWeight;
        }
    }

    float invWeight = 1.0f / weightSum;
    sum = { sum.x * invWeight, sum.y * invWeight, sum.z * invWeight };

    const uchar4* row = (const uchar4*)((const char*)src + gy * srcPitch);

    uchar4 out = float4ToUchar4(sum);
    out.w = row[gx].w;

    uchar4* dstRow = (uchar4*)((char*)dst + gy * dstPitch);
    dstRow[gx] = out;
}

template <int R>
__global__ void BilateralKernelT_Uchar(
    const uchar4* __restrict__ src,
    size_t srcPitch,
    uchar4* __restrict__ dst,
    size_t dstPitch,
    int width,
    int height,
    const float* __restrict__ colorRangeLUT)
{
    extern __shared__ uchar4 bilatTile[];

    const int tileWidth = blockDim.x + 2 * R;
    const int tileHeight = blockDim.y + 2 * R;
    const int blockSize = blockDim.x * blockDim.y;

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    const int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    // last right-bottom block(s) is fully load but not fully used.
    // only if img.dim is fully divisible by blockDim then fully used.
    for (int i = threadId; i < tileWidth * tileHeight; i += blockSize) {
        int tileX = i % tileWidth;
        int tileY = i / tileWidth;

        int srcX = clamp(int(blockIdx.x * blockDim.x) + tileX - R, 0, width - 1);
        int srcY = clamp(int(blockIdx.y * blockDim.y) + tileY - R, 0, height - 1);

        const uchar4* row = (const uchar4*)((const char*)src + srcY * srcPitch);
        uchar4 p = __ldg(&row[srcX]);

        unsigned char lumaPart = unsigned char((77 * p.x + 150 * p.y + 29 * p.z) >> 8);
        bilatTile[i] = make_uchar4(p.x, p.y, p.z, lumaPart);
    }

    __syncthreads();

    if (gx >= width || gy >= height) {
        return;
    }

    float4 sum = {};
    float weightSum = 0.0f;

    int tileCenterX = threadIdx.x + R;
    int tileCenterY = threadIdx.y + R;

    uchar4 center = bilatTile[tileCenterX + tileCenterY * tileWidth];
    constexpr int K = 2 * R + 1;

    #pragma unroll
    for (int ky = 0; ky < K; ++ky) {
        int rowBase = (threadIdx.y + ky) * tileWidth + threadIdx.x;
        int spatialBase = ky * K;

        #pragma unroll
        for (int kx = 0; kx < K; ++kx) {
            uchar4 sample = bilatTile[rowBase + kx];

            float spatialWeight = c_spatial2D[spatialBase + kx];

            int dY = abs(int(center.w) - int(sample.w)); // cast uchar -> int
            float colorWeight = __ldg(&colorRangeLUT[dY]);

            float finalWeight = spatialWeight * colorWeight;

            sum.x += float(sample.x) * finalWeight; // cast uchar -> float
            sum.y += float(sample.y) * finalWeight;
            sum.z += float(sample.z) * finalWeight;

            weightSum += finalWeight;
        }
    }

    float invWeight = 1.0f / weightSum;
    sum = { sum.x * invWeight, sum.y * invWeight, sum.z * invWeight };

    const uchar4* row = (const uchar4*)((const char*)src + gy * srcPitch);

    uchar4 out = float4ToUchar4(sum);
    out.w = row[gx].w;

    uchar4* dstRow = (uchar4*)((char*)dst + gy * dstPitch);
    dstRow[gx] = out;
}

namespace cuda {
    template<int R>
    void PrintOccupancyForBilateral(dim3 block, size_t sharedMemSize) {
        int device = 0;
        cudaGetDevice(&device);

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);

        int activeBlocks = 0;

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks,
            BilateralKernelT_Uchar<R>,     // or BilateralKernelT_Uchar<R>
            block.x * block.y,
            sharedMemSize
        );

        int activeThreads = activeBlocks * block.x * block.y;
        int maxThreads = prop.maxThreadsPerMultiProcessor;

        float occupancy = float(activeThreads) / float(maxThreads);

        std::cout
            << "Block: (" << block.x << ", " << block.y << ")\n"
            << "Requested shared/block: " << sharedMemSize << " bytes\n"
            << "Active blocks per SM: " << activeBlocks << "\n"
            << "Active threads per SM: " << activeThreads << "\n"
            << "Max threads per SM: " << maxThreads << "\n"
            << "Occupancy by threads: " << occupancy * 100.0f << "%\n";
    }
}

namespace filter {
	ImageGPU<uchar4> GaussianBlur(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx) {
        dim3 gridSize = DivUp(input.Dim(), blockSize);

        const auto weights_cpu = gauss::MakeNormGaussWeights(filter_halfsize, sigma);
        cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

        ImageGPU<float4> tmp(input.Dim());
        ImageGPU<uchar4> output(input.Dim());

        cuda::TimedCall("GaussianBlurX+Y: " + util::BlockDimToString(blockSize), ctx, [&]() {
            GaussianBlurX<<<gridSize, vec2Todim3(blockSize), 0, ctx.m_stream>>> (
                input.Data(),
                input.Pitch(),
                tmp.Data(),
                tmp.Pitch(),
                input.Dim().x,
                input.Dim().y,
                filter_halfsize
            );

            GaussianBlurY<<<gridSize, vec2Todim3(blockSize), 0, ctx.m_stream>>> (
                tmp.Data(),
                tmp.Pitch(),
                output.Data(),
                output.Pitch(),
                input.Dim().x,
                input.Dim().y,
                filter_halfsize
            );
        });

        return output;
	}

    // this avoids intermediate write to global mem
    ImageGPU<uchar4> GaussianBlurFusedV1(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx) {
        // TODO: test with diff sizes
        blockSize = { blockSize.x, unsigned int(blockSize.y + filter_halfsize * 2) }; // overlap in Y-dir
        dim3 gridSize = DivUp(input.Dim(), blockSize);

        const auto weights_cpu = gauss::MakeNormGaussWeights(filter_halfsize, sigma);
        cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

        ImageGPU<uchar4> output(input.Dim());
        const size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float4);

        // Correct grid size since useful input in vertical pass are only center w/o halo
        // So need to have more blocks
        const int outputPerBlockY = blockSize.y - 2 * filter_halfsize;
        gridSize.y = (input.Dim().y + outputPerBlockY - 1) / outputPerBlockY;

        cuda::TimedCall("GaussianBlurTileV1: " + util::BlockDimToString(blockSize), ctx, [&]() {
            GaussianBlurTile<<<gridSize, vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                input.Data(),
                input.Pitch(),
                output.Data(),
                output.Pitch(),
                input.Dim().x,
                input.Dim().y,
                filter_halfsize
            );
        });

        return output;
    }

    ImageGPU<uchar4> GaussianBlurFusedV2(const ImageGPU<uchar4>& input, int filter_halfsize, float sigma, vec2ui blockSize, cuda::KernelContext& ctx) {
        // TODO: test with diff sizes
        const auto weights_cpu = gauss::MakeNormGaussWeights(filter_halfsize, sigma);
        cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

        ImageGPU<uchar4> output(input.Dim());
        const size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float4);

        // Correct grid size since useful input in vertical pass are only center w/o halo
        // So need to have more blocks
        vec2ui outputBlock = { blockSize.x, blockSize.y - 2 * filter_halfsize };
        dim3 gridSize = DivUp(input.Dim(), outputBlock);

        cuda::TimedCall("GaussianBlurTileV2: " + util::BlockDimToString(blockSize), ctx, [&]() {
            GaussianBlurTile<<<gridSize, vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                input.Data(),
                input.Pitch(),
                output.Data(),
                output.Pitch(),
                input.Dim().x,
                input.Dim().y,
                filter_halfsize
            );
        });

        return output;
    }

    ImageGPU<uchar4> BilateralFilter(const ImageGPU<uchar4>& input, int filter_halfsize, float sigmaGauss, float sigmaColor, vec2ui blockSize, cuda::KernelContext& ctx) {
        dim3 gridSize = DivUp(input.Dim(), blockSize);

        // can use non-norm weights, don't matter
        const auto weights_cpu = gauss::Make2dGaussWeights(filter_halfsize, sigmaGauss);
        cudaCheck(cudaMemcpyToSymbolAsync(c_spatial2D, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

        const auto range_color_lut = bilateral::RangeLUT(sigmaColor);
        float* colorRangeLUT = common::AllocAndCopyCPU(range_color_lut, ctx.m_stream);

        ImageGPU<uchar4> output(input.Dim());

        const size_t tileWidth = blockSize.x + filter_halfsize * 2;
        const size_t tileHeight = blockSize.y + filter_halfsize * 2;

        //const size_t tileStride = tileWidth;
        const size_t sharedMemSize = tileWidth * tileHeight * sizeof(int4);

        cuda::TimedCall("BilateralKernel: " + util::BlockDimToString(blockSize), ctx, [&]() {
            BilateralKernel<<<gridSize, vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                input.Data(),
                input.Pitch(),
                output.Data(),
                output.Pitch(),
                input.Dim().x,
                input.Dim().y,
                filter_halfsize,
                colorRangeLUT
            );
        });

        cudaFreeAsync(colorRangeLUT, ctx.m_stream);
        return output;
    }

    ImageGPU<uchar4> BilateralFilterT(const ImageGPU<uchar4>& input, float sigmaGauss, float sigmaColor, vec2ui blockSize, cuda::KernelContext& ctx, bool debugInfo) {
        dim3 gridSize = DivUp(input.Dim(), blockSize);

        constexpr int filter_halfsize = 5;

        // can use non-norm weights, don't matter
        const auto weights_cpu = gauss::Make2dGaussWeights(filter_halfsize, sigmaGauss);
        cudaCheck(cudaMemcpyToSymbolAsync(c_spatial2D, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

        const auto range_color_lut = bilateral::RangeLUT(sigmaColor);
        float* colorRangeLUT = common::AllocAndCopyCPU(range_color_lut, ctx.m_stream);

        ImageGPU<uchar4> output(input.Dim());

        const size_t tileWidth = blockSize.x + filter_halfsize * 2;
        const size_t tileHeight = blockSize.y + filter_halfsize * 2;

        const auto gpuArch = gpu::DetectArch();
        if (gpuArch == gpu::Arch::Unknown) {
            throw std::logic_error("gpu::Arch is unknown");
        }

        const size_t typeSize = (gpuArch == gpu::Arch::Turing) ? sizeof(uchar4) : sizeof(int4);
        const size_t sharedMemSize = tileWidth * tileHeight * typeSize;

        if (debugInfo) {
            cuda::PrintOccupancyForBilateral<filter_halfsize>(vec2Todim3(blockSize), sharedMemSize);
            std::cout << std::endl;
        }

        // Quadro T1000/T2000
        if (gpuArch == gpu::Arch::Turing) {
            cuda::TimedCall("BilateralKernel_Uchar4: " + util::BlockDimToString(blockSize), ctx, [&]() {
                BilateralKernelT_Uchar<filter_halfsize><<<gridSize, vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                    input.Data(),
                    input.Pitch(),
                    output.Data(),
                    output.Pitch(),
                    input.Dim().x,
                    input.Dim().y,
                    colorRangeLUT
                );
            });
        }
        // RTX and others
        // Very surpisingly but casts from uchar -> int/float are hurting more than wasting memory.
        // This is opposite on Quadro.
        // Idk if this is better on rtx 2/3, tested on rtx 4050 and rtx 5060.
        else {
            cuda::TimedCall("BilateralKernel_Int4: " + util::BlockDimToString(blockSize), ctx, [&]() {
                BilateralKernelT<filter_halfsize><<<gridSize, vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                    input.Data(),
                    input.Pitch(),
                    output.Data(),
                    output.Pitch(),
                    input.Dim().x,
                    input.Dim().y,
                    colorRangeLUT
                );
            });
        }

        cudaFreeAsync(colorRangeLUT, ctx.m_stream);
        return output;
    }
}
