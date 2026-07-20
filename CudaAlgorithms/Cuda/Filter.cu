#include "Filter.h"
#include "Common.cuh"
#include "KernelCommon.h"
#include "../Common/Common.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

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

    float4 sum = {};

    const uchar4* srcRow = (const uchar4*)((const char*)src + y * srcPitch);
    for (int dx = -filter_halfsize; dx <= filter_halfsize; ++dx) {
        int idx = clamp(x + dx, 0, width - 1);

        uchar4 p = srcRow[idx];
        float4 sample = { float(p.x), float(p.y), float(p.z), 0.0f };

        sum.x += (sample.x * c_weightsGauss[abs(dx)]);
        sum.y += (sample.y * c_weightsGauss[abs(dx)]);
        sum.z += (sample.z * c_weightsGauss[abs(dx)]);
    }

    float4* dstRow = (float4*)((char*)dst + y * dstPitch);
    dstRow[x] = { sum.x, sum.y, sum.z, float(srcRow[x].w) };
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

    float4 sum = {};

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

    uchar4 output = floatVecToUchar(sum);
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
            float4 sample = { float(p.x), float(p.y), float(p.z), 0.0f };

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

        uchar4 result = floatVecToUchar(sum);
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
        uchar4 p = row[srcX];

        int lumaPart = (77 * p.x + 150 * p.y + 29 * p.z) >> 8;
        bilateralTile[i] = { p.x, p.y, p.z, lumaPart };
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

    uchar4 out = floatVecToUchar(sum);
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
        uchar4 p = row[srcX];

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

    uchar4 out = floatVecToUchar(sum);
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
        uchar4 p = row[srcX];

        unsigned char lumaPart = unsigned char((77 * p.x + 150 * p.y + 29 * p.z) >> 8);
        bilatTile[i] = { p.x, p.y, p.z, lumaPart };
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

    uchar4 out = floatVecToUchar(sum);
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

namespace cuda {
    namespace filter {
        void GaussianBlur(
            const image::GpuImageView<uchar4>& input,
            image::GpuImageView<float4>& tmp_buffer,
            image::GpuImageView<uchar4>& output,
            const GaussianBlurParams& p,
            cuda::KernelContext& ctx)
        {
            dim3 gridSize = cuda::math::DivUp(input.m_dim, p.blockDim);

            const auto weights_cpu = gauss::MakeNormGaussWeights(p.filter_halfsize, p.sigma);
            cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

            cuda::TimedCall("GaussianBlurX+Y", ctx, [&]() {
                GaussianBlurX<<<gridSize, cuda::math::vec2Todim3(p.blockDim), 0, ctx.m_stream>>> (
                    input.m_ptr,
                    input.m_pitch,
                    tmp_buffer.m_ptr,
                    tmp_buffer.m_pitch,
                    input.m_dim.x,
                    input.m_dim.y,
                    p.filter_halfsize
                );

                GaussianBlurY<<<gridSize, cuda::math::vec2Todim3(p.blockDim), 0, ctx.m_stream>>> (
                    tmp_buffer.m_ptr,
                    tmp_buffer.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    input.m_dim.x,
                    input.m_dim.y,
                    p.filter_halfsize
                );
            });
        }

        // this avoids intermediate write to global mem
        void GaussianBlurFusedV1(
            const image::GpuImageView<uchar4>& input,
            image::GpuImageView<uchar4>& output,
            const GaussianBlurParams& p,
            cuda::KernelContext& ctx)
        {
            image::vec2ui blockSize = { p.blockDim.x, unsigned int(p.blockDim.y + p.filter_halfsize * 2) }; // overlap in Y-dir
            dim3 gridSize = cuda::math::DivUp(input.m_dim, blockSize);

            const auto weights_cpu = gauss::MakeNormGaussWeights(p.filter_halfsize, p.sigma);
            cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

            const size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float4);

            // Correct grid size since useful input in vertical pass are only center w/o halo
            // So need to have more blocks
            const int outputPerBlockY = blockSize.y - 2 * p.filter_halfsize;
            gridSize.y = (input.m_dim.y + outputPerBlockY - 1) / outputPerBlockY;

            cuda::TimedCall("GaussianBlurTileV1", ctx, [&]() {
                GaussianBlurTile<<<gridSize, cuda::math::vec2Todim3(blockSize), sharedMemSize, ctx.m_stream>>> (
                    input.m_ptr,
                    input.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    input.m_dim.x,
                    input.m_dim.y,
                    p.filter_halfsize
                );
            });
        }

        void GaussianBlurFusedV2(
            const image::GpuImageView<uchar4>& input,
            image::GpuImageView<uchar4>& output,
            const GaussianBlurParams& p,
            cuda::KernelContext& ctx)
        {
            const auto weights_cpu = gauss::MakeNormGaussWeights(p.filter_halfsize, p.sigma);
            cudaCheck(cudaMemcpyToSymbolAsync(c_weightsGauss, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

            const size_t sharedMemSize = p.blockDim.x * p.blockDim.y * sizeof(float4);

            // Correct grid size since useful input in vertical pass are only center w/o halo
            // So need to have more blocks
            image::vec2ui outputBlock = { p.blockDim.x, p.blockDim.y - 2 * p.filter_halfsize };
            dim3 gridSize = cuda::math::DivUp(input.m_dim, outputBlock);

            cuda::TimedCall("GaussianBlurTileV2", ctx, [&]() {
                GaussianBlurTile<<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                    input.m_ptr,
                    input.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    input.m_dim.x,
                    input.m_dim.y,
                    p.filter_halfsize
                );
            });
        }

        void BilateralFilter(
            const image::GpuImageView<uchar4>& input,
            image::GpuImageView<uchar4>& output,
            const BilateralParams& p,
            cuda::KernelContext& ctx)
        {
            dim3 gridSize = cuda::math::DivUp(input.m_dim, p.blockDim);

            // can use non-norm weights, don't matter
            const auto weights_cpu = gauss::Make2dGaussWeights(p.filter_halfsize, p.sigmaGauss);
            cudaCheck(cudaMemcpyToSymbolAsync(c_spatial2D, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

            const auto range_color_lut = bilateral::RangeLUT(p.sigmaColor);
            float* colorRangeLUT = common::AllocAndCopyCPU(range_color_lut, ctx.m_stream);

            const size_t tileWidth = p.blockDim.x + p.filter_halfsize * 2;
            const size_t tileHeight = p.blockDim.y + p.filter_halfsize * 2;

            const size_t sharedMemSize = tileWidth * tileHeight * sizeof(int4);

            cuda::TimedCall("BilateralKernel", ctx, [&]() {
                BilateralKernel<<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                    input.m_ptr,
                    input.m_pitch,
                    output.m_ptr,
                    output.m_pitch,
                    input.m_dim.x,
                    input.m_dim.y,
                    p.filter_halfsize,
                    colorRangeLUT
                );
            });

            cudaFreeAsync(colorRangeLUT, ctx.m_stream);
        }

        void BilateralFilterT(
            const image::GpuImageView<uchar4>& input,
            image::GpuImageView<uchar4>& output,
            const BilateralParams& p,
            cuda::KernelContext& ctx,
            bool debugInfo)
        {
            dim3 gridSize = cuda::math::DivUp(input.m_dim, p.blockDim);

            constexpr int filter_halfsize = 5;

            // can use non-norm weights, don't matter
            const auto weights_cpu = gauss::Make2dGaussWeights(filter_halfsize, p.sigmaGauss);
            cudaCheck(cudaMemcpyToSymbolAsync(c_spatial2D, weights_cpu.data(), weights_cpu.size() * sizeof(float), 0, cudaMemcpyHostToDevice, ctx.m_stream));

            const auto range_color_lut = bilateral::RangeLUT(p.sigmaColor);
            float* colorRangeLUT = common::AllocAndCopyCPU(range_color_lut, ctx.m_stream);

            const size_t tileWidth = p.blockDim.x + filter_halfsize * 2;
            const size_t tileHeight = p.blockDim.y + filter_halfsize * 2;

            // TODO: maybe extend for other kernels
            const auto gpuArch = cuda::gpu::DetectArch();
            if (gpuArch == cuda::gpu::Arch::Unknown) {
                throw std::logic_error("gpu::Arch is unknown");
            }

            const size_t typeSize = (gpuArch == cuda::gpu::Arch::Turing) ? sizeof(uchar4) : sizeof(int4);
            const size_t sharedMemSize = tileWidth * tileHeight * typeSize;

            if (debugInfo) {
                cuda::PrintOccupancyForBilateral<filter_halfsize>(cuda::math::vec2Todim3(p.blockDim), sharedMemSize);
                std::cout << std::endl;
            }

            // Quadro T1000/T2000
            if (gpuArch == cuda::gpu::Arch::Turing) {
                cuda::TimedCall("BilateralKernel_Uchar4", ctx, [&]() {
                    BilateralKernelT_Uchar<filter_halfsize> <<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                        input.m_ptr,
                        input.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        input.m_dim.x,
                        input.m_dim.y,
                        colorRangeLUT
                    );
                });
            }
            // RTX and others
            // Very surpisingly but casts from uchar -> int/float are hurting more than wasting memory.
            // This is opposite on Quadro.
            // Idk if this is better on rtx 2/3, tested on rtx 4050 and rtx 5060.
            else {
                cuda::TimedCall("BilateralKernel_Int4", ctx, [&]() {
                    BilateralKernelT<filter_halfsize> <<<gridSize, cuda::math::vec2Todim3(p.blockDim), sharedMemSize, ctx.m_stream>>> (
                        input.m_ptr,
                        input.m_pitch,
                        output.m_ptr,
                        output.m_pitch,
                        input.m_dim.x,
                        input.m_dim.y,
                        colorRangeLUT
                    );
                });
            }

            cudaFreeAsync(colorRangeLUT, ctx.m_stream);
        }
    }
}
