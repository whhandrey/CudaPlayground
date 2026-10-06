#include "Sobel.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <vector>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__  void SobelMag3dNaiveSharedMemKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ sobelX,
	const float* __restrict__ sobelY,
	const float* __restrict__ sobelZ,
	int3 dim)
{
	extern __shared__ float tile[];

	const int3 filter_halfsize = { 1, 1, 1 };

	const int tileWidth = blockDim.x + filter_halfsize.x * 2;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;
	const int tileDepth = blockDim.z + filter_halfsize.z * 2;

	const int tileSize = tileWidth * tileHeight * tileDepth;
	const int blockSize = blockDim.x * blockDim.y * blockDim.z;

	const int3 blockOrigin = {
		static_cast<int>(blockIdx.x * blockDim.x),
		static_cast<int>(blockIdx.y * blockDim.y),
		static_cast<int>(blockIdx.z * blockDim.z)
	};

	const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

	for (int i = tid; i < tileSize; i += blockSize) {
		const int tileZ = i / (tileWidth * tileHeight);
		const int tileXY = i % (tileWidth * tileHeight);

		const int tileY = tileXY / tileWidth;
		const int tileX = tileXY % tileWidth;

		const int sampleX = clamp(blockOrigin.x + tileX - filter_halfsize.x, 0, dim.x - 1);
		const int sampleY = clamp(blockOrigin.y + tileY - filter_halfsize.y, 0, dim.y - 1);
		const int sampleZ = clamp(blockOrigin.z + tileZ - filter_halfsize.z, 0, dim.z - 1);

		const size_t offsetBytes = OffsetToRowPitched(sampleY, sampleZ, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytes);

		tile[i] = row[sampleX];
	}

	__syncthreads();

	float gx = 0.0f;
	float gy = 0.0f;
	float gz = 0.0f;

	const int x = blockOrigin.x + threadIdx.x;
	const int y = blockOrigin.y + threadIdx.y;
	const int z = blockOrigin.z + threadIdx.z;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	const int filterWidth = filter_halfsize.x * 2 + 1;
	const int filterHeight = filter_halfsize.y * 2 + 1;

	const int filterArea = filterWidth * filterHeight;
	const int tileArea = tileWidth * tileHeight;

	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; dz++) {
		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; dy++) {
			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; dx++) {

				const int tileIndex = (threadIdx.x + dx + filter_halfsize.x)
					+ (threadIdx.y + dy + filter_halfsize.y) * tileWidth
					+ (threadIdx.z + dz + filter_halfsize.z) * tileArea;

				const int coeffIndex = (dx + filter_halfsize.x)
					+ (dy + filter_halfsize.y) * filterWidth
					+ (dz + filter_halfsize.z) * filterArea;

				gx += tile[tileIndex] * sobelX[coeffIndex];
				gy += tile[tileIndex] * sobelY[coeffIndex];
				gz += tile[tileIndex] * sobelZ[coeffIndex];
			}
		}
	}

	const float gradMag = sqrtf(gx * gx + gy * gy + gz * gz);

	const size_t offsetBytes = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
	float* row = (float*)((char*)output + offsetBytes);

	row[x] = gradMag;
}

// S = { 0.25f, 0.5f, 0.25f }
__device__ __forceinline__ float SobelSmooth(float left, float center, float right) {
	return 0.25f * (left + right) + 0.5f * center;
}

// D = { -0.5f, 0.0f, 0.5f }
__device__ __forceinline__ float SobelDerivative(float left, float right) {
	return 0.5f * (right - left);
}

struct SubPlane3x3 {
	float v00, v01, v02;
	float v10, v11, v12;
	float v20, v21, v22;
};

__device__ __forceinline__ SubPlane3x3 ReadSubPlaneFromTile(
	const float* __restrict__ tile,
	int centerX,
	int centerY,
	int tileWidth)
{
	int center = centerX + centerY * tileWidth;

	return {
		tile[center - tileWidth - 1],
		tile[center - tileWidth],
		tile[center - tileWidth + 1],

		tile[center - 1],
		tile[center],
		tile[center + 1],

		tile[center + tileWidth - 1],
		tile[center + tileWidth],
		tile[center + tileWidth + 1]
	};
}

struct SobelPlane
{
	float dxSy; // derivative X, smoothing Y
	float sxDy; // smoothing X, derivative Y
	float sxSy; // smoothing X, smoothing Y
};

__device__ __forceinline__ SobelPlane CalculateSobelPlane(const SubPlane3x3& p) {
	// Process every row in the X direction.
	const float sx0 = SobelSmooth(p.v00, p.v01, p.v02);
	const float sx1 = SobelSmooth(p.v10, p.v11, p.v12);
	const float sx2 = SobelSmooth(p.v20, p.v21, p.v22);

	const float dx0 = SobelDerivative(p.v00, p.v02);
	const float dx1 = SobelDerivative(p.v10, p.v12);
	const float dx2 = SobelDerivative(p.v20, p.v22);

	return {
		// Dx followed by Sy
		SobelSmooth(dx0, dx1, dx2),

		// Sx followed by Dy
		SobelDerivative(sx0, sx2),

		// Sx followed by Sy
		SobelSmooth(sx0, sx1, sx2)
	};
}

__global__  void SobelMag3dSeparableFusedKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	int3 dim)
{
	extern __shared__ float tile[];

	constexpr int3 filter_halfsize = { 1, 1, 1 };

	const int tileWidth = blockDim.x + filter_halfsize.x * 2;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;
	const int tileDepth = blockDim.z + filter_halfsize.z * 2;

	const int tileSize = tileWidth * tileHeight * tileDepth;
	const int tilePlaneSize = tileWidth * tileHeight;

	const int blockSize = blockDim.x * blockDim.y * blockDim.z;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z * blockDim.z;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

	// cooperative tile load into shared mem
	for (int i = tid; i < tileSize; i += blockSize) {
		const int tileZ = i / tilePlaneSize;
		const int tileXY = i % tilePlaneSize;

		const int tileY = tileXY / tileWidth;
		const int tileX = tileXY % tileWidth;

		int x = clamp(originX + tileX - filter_halfsize.x, 0, dim.x - 1);
		int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);
		int z = clamp(originZ + tileZ - filter_halfsize.z, 0, dim.z - 1);

		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		tile[i] = row[x];
	}

	__syncthreads();

	const int centerX = threadIdx.x + filter_halfsize.x;
	const int centerY = threadIdx.y + filter_halfsize.y;

	SobelPlane planes[3];
	for (int i = 0; i < 3; ++i) {
		const int tileZ = threadIdx.z + i;
		const float* tilePlane = tile + tileZ * tilePlaneSize;

		const SubPlane3x3 subPlane = ReadSubPlaneFromTile(tilePlane, centerX, centerY, tileWidth);
		const SobelPlane plane = CalculateSobelPlane(subPlane);

		planes[i] = plane;
	}

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y);
	const int z = originZ + int(threadIdx.z);

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	const float gx = SobelSmooth(planes[0].dxSy, planes[1].dxSy, planes[2].dxSy);
	const float gy = SobelSmooth(planes[0].sxDy, planes[1].sxDy, planes[2].sxDy);
	const float gz = SobelDerivative(planes[0].sxSy, planes[2].sxSy);

	const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
	float* row = (float*)((char*)output + offsetBytesToRow);

	const float magnitude = sqrtf(gx * gx + gy * gy + gz * gz);
	row[x] = magnitude;
}

__device__ __forceinline__ SobelPlane CalculatePlane(
	const float* __restrict__ tile,
	int centerX,
	int centerY,
	int tileZ,
	int tileWidth,
	int tilePlaneSize)
{
	const float* tilePlane = tile + tileZ * tilePlaneSize;
	const SubPlane3x3 subPlane = ReadSubPlaneFromTile(tilePlane, centerX, centerY, tileWidth);

	return CalculateSobelPlane(subPlane);
}

template <int outputsPerThread>
__global__  void SobelMag3dSepFusedMultipleOutputsKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	int3 dim)
{
	extern __shared__ float tile[];

	constexpr int3 filter_halfsize = { 1, 1, 1 };
	const int outputDepth = blockDim.z * outputsPerThread;

	const int tileWidth = blockDim.x + filter_halfsize.x * 2;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;
	const int tileDepth = outputDepth + filter_halfsize.z * 2;

	const int tileSize = tileWidth * tileHeight * tileDepth;
	const int tilePlaneSize = tileWidth * tileHeight;

	const int blockSize = blockDim.x * blockDim.y * blockDim.z;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z * outputDepth;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

	// cooperative tile load into shared mem
	for (int i = tid; i < tileSize; i += blockSize) {
		const int tileZ = i / tilePlaneSize;
		const int tileXY = i % tilePlaneSize;

		const int tileY = tileXY / tileWidth;
		const int tileX = tileXY % tileWidth;

		int x = clamp(originX + tileX - filter_halfsize.x, 0, dim.x - 1);
		int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);
		int z = clamp(originZ + tileZ - filter_halfsize.z, 0, dim.z - 1);

		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		tile[i] = row[x];
	}

	__syncthreads();

	const int centerX = threadIdx.x + filter_halfsize.x;
	const int centerY = threadIdx.y + filter_halfsize.y;

	const int x = originX + threadIdx.x;
	const int y = originY + threadIdx.y;
	const int firstOutputZ = originZ + threadIdx.z * outputsPerThread;

	if (x >= dim.x || y >= dim.y || firstOutputZ >= dim.z) {
		return;
	}

	SobelPlane prev = CalculatePlane(tile, centerX, centerY, threadIdx.z * outputsPerThread, tileWidth, tilePlaneSize);
	SobelPlane curr = CalculatePlane(tile, centerX, centerY, threadIdx.z * outputsPerThread + 1, tileWidth, tilePlaneSize);
	SobelPlane next;

	for (int i = 0; i < outputsPerThread; ++i) {
		next = CalculatePlane(tile, centerX, centerY, threadIdx.z * outputsPerThread + (i + 2), tileWidth, tilePlaneSize);
		const int outputZ = firstOutputZ + i;

		// All following Z positions will also be outside.
		if (outputZ >= dim.z) {
			return;
		}

		const float gx = SobelSmooth(prev.dxSy, curr.dxSy, next.dxSy);
		const float gy = SobelSmooth(prev.sxDy, curr.sxDy, next.sxDy);
		const float gz = SobelDerivative(prev.sxSy, next.sxSy);

		const size_t offsetBytesToRow = OffsetToRowPitched(y, outputZ, outPitch, outSlicePitch);
		float* row = (float*)((char*)output + offsetBytesToRow);

		const float magnitude = sqrtf(gx * gx + gy * gy + gz * gz);
		row[x] = magnitude;

		prev = curr;
		curr = next;
	}
}

namespace cuda {
	namespace grad3d {
		void SobelMagNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> sobelX,
			memory::GpuSpan<const float> sobelY,
			memory::GpuSpan<const float> sobelZ,
			image::GpuVolumeView<float> output,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			constexpr size_t sobelSize = 3 * 3 * 3;

			if (sobelX.m_size != sobelSize) {
				throw std::logic_error("CudaAlgoritms::SobelMagNaiveSharedMem: invalid sobelX size");
			}

			if (sobelY.m_size != sobelSize) {
				throw std::logic_error("CudaAlgoritms::SobelMagNaiveSharedMem: invalid sobelY size");
			}

			if (sobelZ.m_size != sobelSize) {
				throw std::logic_error("CudaAlgoritms::SobelMagNaiveSharedMem: invalid sobelZ size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::SobelMagNaiveSharedMem: input/output dim mismatch");
			}

			const image::vec3ui filter_halfsize{ 1, 1, 1 };

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
			const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * tileDepth * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("SobelMag3dNaiveSharedMemKernel", ctx, [&]() {
				SobelMag3dNaiveSharedMemKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					sobelX.m_ptr,
					sobelY.m_ptr,
					sobelZ.m_ptr,
					dim
				);
			});
		}

		void SobelMagFusedSeparable(image::GpuVolumeView<const float> input, image::GpuVolumeView<float> output, cuda::KernelContext ctx, image::vec3ui blockDim) {
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::SobelMagFusedSeparable: input/output dim mismatch");
			}

			const image::vec3ui filter_halfsize{ 1, 1, 1 };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const size_t tileWidth = blockDim.x + filter_halfsize.x * 2;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;
			const size_t tileDepth = blockDim.z + filter_halfsize.z * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * tileDepth * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("SobelMag3dSeparableFusedKernel", ctx, [&]() {
				SobelMag3dSeparableFusedKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					dim
				);
			});
		}

		void SobelMagFusedSeparableMultipleOutputs(image::GpuVolumeView<const float> input, image::GpuVolumeView<float> output, int outputsPerThread, cuda::KernelContext ctx, image::vec3ui blockDim) {
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::SobelMagFusedSeparableMultipleOutputs: input/output dim mismatch");
			}

			if (outputsPerThread < 2 || outputsPerThread > 10) {
				throw std::logic_error("CudaAlgoritms::SobelMagFusedSeparableMultipleOutputs: unsupported outputsPerThread");
			}

			const image::vec3ui filter_halfsize{ 1, 1, 1 };
			
			const auto blockDimGrid = image::vec3ui{ blockDim.x, blockDim.y, blockDim.z * unsigned int(outputsPerThread) };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);

			const size_t tileWidth = blockDimGrid.x + filter_halfsize.x * 2;
			const size_t tileHeight = blockDimGrid.y + filter_halfsize.y * 2;
			const size_t tileDepth = blockDimGrid.z + filter_halfsize.z * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * tileDepth * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 2) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<2>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<3>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<4>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<4> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<5>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<6>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<6> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<7>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<7> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<8>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<8> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<9>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<9> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("SobelMag3dSepFusedMulOutsKernel<10>", ctx, [&]() {
					SobelMag3dSepFusedMultipleOutputsKernel<10> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						dim
					);
				});
			}
		}
	}
}
