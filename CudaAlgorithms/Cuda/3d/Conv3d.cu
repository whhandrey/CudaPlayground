#include "Conv3d.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <stdexcept>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__  void Conv3dNaiveKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffs3d,
	image::vec3i filter_halfsize,
	int3 dim)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	const int filterWidth = filter_halfsize.x * 2 + 1;
	const int filterArea = filterWidth * (filter_halfsize.y * 2 + 1);

	float sampleOut = 0.0f;

	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; dz++) {
		const int z_idx = clamp(z + dz, 0, dim.z - 1);

		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; dy++) {
			const int y_idx = clamp(y + dy, 0, dim.y - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y_idx, z_idx, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; dx++) {
				const int x_idx = clamp(x + dx, 0, dim.x - 1);

				const int coeffIndex = (dx + filter_halfsize.x)
					+ (dy + filter_halfsize.y) * filterWidth
					+ (dz + filter_halfsize.z) * filterArea;

				sampleOut += row[x_idx] * coeffs3d[coeffIndex];
			}
		}
	}

	const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);

	float* row = (float*)((char*)output + offsetBytesToRow);
	row[x] = sampleOut;
}

__global__  void Conv3dNaiveSharedMemKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffs3d,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

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

		const size_t offsetBytesToRow = OffsetToRowPitched(sampleY, sampleZ, inPitch, inSlicePitch);

		const float* row = (const float*)((const char*)input + offsetBytesToRow);
		tile[i] = row[sampleX];
	}

	__syncthreads();

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

	float sampleOut = 0.0f;

	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; dz++) {
		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; dy++) {
			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; dx++) {

				const int tileIndex = (threadIdx.x + dx + filter_halfsize.x)
					+ (threadIdx.y + dy + filter_halfsize.y) * tileWidth
					+ (threadIdx.z + dz + filter_halfsize.z) * tileArea;

				const int coeffIndex = (dx + filter_halfsize.x)
					+ (dy + filter_halfsize.y) * filterWidth
					+ (dz + filter_halfsize.z) * filterArea;

				sampleOut += tile[tileIndex] * coeffs3d[coeffIndex];
			}
		}
	}

	const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);

	float* row = (float*)((char*)output + offsetBytesToRow);
	row[x] = sampleOut;
}

__global__  void Conv3dSeparableFusedKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffsX,
	const float* __restrict__ coeffsY,
	const float* __restrict__ coeffsZ,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

	const int tileWidth = blockDim.x;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;

	const int tileSize = tileWidth * tileHeight;
	const int blockSize = blockDim.x * blockDim.y;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	float result = 0.0f;
	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; ++dz) {
		int z = clamp(originZ + dz, 0, dim.z - 1);

		// x-pass without shared mem using cooperative convolution (to include halo rows for y pass)
		for (int i = tid; i < tileSize; i += blockSize) {
			const int tileX = i % tileWidth;
			const int tileY = i / tileWidth;

			const int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			float convX = 0.0f;
			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
				int x = clamp(originX + tileX + dx, 0, dim.x - 1);
				convX += row[x] * coeffsX[dx + filter_halfsize.x];
			}

			tile[i] = convX;
		}

		__syncthreads();

		const int tileX = threadIdx.x;
		const int tileY = threadIdx.y + filter_halfsize.y;

		float convY = 0.0f;
		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			convY += tile[tileX + (tileY + dy) * tileWidth] * coeffsY[dy + filter_halfsize.y];
		}

		// ensure you don't override tile on the next iteration while some warp is still reading data in y-pass
		// on last iteration not necessary
		if (dz < filter_halfsize.z) {
			__syncthreads();
		}

		result += convY * coeffsZ[dz + filter_halfsize.z];
	}

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y);
	const int z = originZ;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
	float* row = (float*)((char*)output + offsetBytesToRow);

	row[x] = result;
}

template <int outputsPerThread, int radiusX, int radiusY, int radiusZ>
__global__  void Conv3dSeparableFusedMultipleOutKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffsX,
	const float* __restrict__ coeffsY,
	const float* __restrict__ coeffsZ,
	int3 dim)
{
	extern __shared__ float tile[];

	const int tileWidth = blockDim.x;
	const int tileHeight = blockDim.y + radiusY * 2;

	const int tileSize = tileWidth * tileHeight;
	const int blockSize = blockDim.x * blockDim.y;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int baseZ = blockIdx.z * outputsPerThread;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	float results[outputsPerThread] = {};
	for (int p = -radiusZ; p < radiusZ + outputsPerThread; ++p) {
		int z = clamp(baseZ + p, 0, dim.z - 1);

		// x-pass without shared mem using cooperative convolution (to include halo rows for y pass)
		for (int i = tid; i < tileSize; i += blockSize) {
			const int tileX = i % tileWidth;
			const int tileY = i / tileWidth;

			const int y = clamp(originY + tileY - radiusY, 0, dim.y - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			float convX = 0.0f;
			for (int dx = -radiusX; dx <= radiusX; ++dx) {
				int x = clamp(originX + tileX + dx, 0, dim.x - 1);
				convX += row[x] * coeffsX[dx + radiusX];
			}

			tile[i] = convX;
		}

		__syncthreads();

		const int tileX = threadIdx.x;
		const int tileY = threadIdx.y + radiusY;

		float convY = 0.0f;
		for (int dy = -radiusY; dy <= radiusY; ++dy) {
			convY += tile[tileX + (tileY + dy) * tileWidth] * coeffsY[dy + radiusY];
		}

		// ensure you don't override tile on the next iteration while some warp is still reading data in y-pass
		// on last iteration not necessary
		if (p < radiusZ + outputsPerThread - 1) {
			__syncthreads();
		}

		for (int i = 0; i < outputsPerThread; ++i) {
			int relativeZ = p - i;
			if (relativeZ >= -radiusZ && relativeZ <= radiusZ) {
				results[i] += convY * coeffsZ[relativeZ + radiusZ];
			}
		}
	}

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y);
	const int z = baseZ;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		if (baseZ + i < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, baseZ + i, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = results[i];
		}
	}
}

template <int outputsPerThread, int planesPerBatch>
__global__  void Conv3dSeparableFusedMultOutBatchKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffsX,
	const float* __restrict__ coeffsY,
	const float* __restrict__ coeffsZ,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

	const int tileWidth = blockDim.x;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;

	const int blockSize = blockDim.x * blockDim.y;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z * outputsPerThread;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	float results[outputsPerThread] = {};
	const int planeEnd = filter_halfsize.z + outputsPerThread;

	for (int p = -filter_halfsize.z; p < filter_halfsize.z + outputsPerThread; p += planesPerBatch) {
		const int tileDepth = min(planeEnd - p, planesPerBatch);
		const int tileSize = tileWidth * tileHeight * tileDepth;

		// x-pass without shared mem using cooperative convolution (to include halo rows for y pass)
		for (int i = tid; i < tileSize; i += blockSize) {
			const int tileZ = i / (tileWidth * tileHeight);
			const int tileXY = i % (tileWidth * tileHeight);

			const int tileY = tileXY / tileWidth;
			const int tileX = tileXY % tileWidth;

			const int z = clamp(originZ + tileZ + p, 0, dim.z - 1);
			const int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			float convX = 0.0f;
			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
				int x = clamp(originX + tileX + dx, 0, dim.x - 1);
				convX += row[x] * coeffsX[dx + filter_halfsize.x];
			}

			tile[i] = convX;
		}

		__syncthreads();

		const int tileX = threadIdx.x;
		const int tileY = threadIdx.y + filter_halfsize.y;

		for (int z = 0; z < tileDepth; ++z) {
			float convY = 0.0f;

			for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
				const int flatIdx = FlatIdx(tileX, tileY + dy, z, tileWidth, tileHeight);
				convY += tile[flatIdx] * coeffsY[dy + filter_halfsize.y];
			}

			for (int i = 0; i < outputsPerThread; ++i) {
				int relativeZ = p + z - i;
				if (relativeZ >= -filter_halfsize.z && relativeZ <= filter_halfsize.z) {
					results[i] += convY * coeffsZ[relativeZ + filter_halfsize.z];
				}
			}
		}

		// ensure you don't override tile on the next iteration while some warp is still reading data in y-pass
		// on last iteration not necessary
		if (p + planesPerBatch < filter_halfsize.z + outputsPerThread) {
			__syncthreads();
		}
	}

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y);
	const int z = originZ;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		if (originZ + i < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, originZ + i, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = results[i];
		}
	}
}

template <int outputsPerThread, int radiusX, int radiusY, int radiusZ>
__global__  void Conv3dSepFusedMultOutWithSigmKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* __restrict__ coeffsX,
	const float* __restrict__ coeffsY,
	const float* __restrict__ coeffsZ,
	bool runSigmoid,
	int3 dim)
{
	extern __shared__ float tile[];

	const int tileWidth = blockDim.x;
	const int tileHeight = blockDim.y + radiusY * 2;

	const int tileSize = tileWidth * tileHeight;
	const int blockSize = blockDim.x * blockDim.y;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int baseZ = blockIdx.z * outputsPerThread;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	float results[outputsPerThread] = {};
	for (int p = -radiusZ; p < radiusZ + outputsPerThread; ++p) {
		int z = clamp(baseZ + p, 0, dim.z - 1);

		// x-pass without shared mem using cooperative convolution (to include halo rows for y pass)
		for (int i = tid; i < tileSize; i += blockSize) {
			const int tileX = i % tileWidth;
			const int tileY = i / tileWidth;

			const int y = clamp(originY + tileY - radiusY, 0, dim.y - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			float convX = 0.0f;
			for (int dx = -radiusX; dx <= radiusX; ++dx) {
				int x = clamp(originX + tileX + dx, 0, dim.x - 1);
				convX += row[x] * coeffsX[dx + radiusX];
			}

			tile[i] = convX;
		}

		__syncthreads();

		const int tileX = threadIdx.x;
		const int tileY = threadIdx.y + radiusY;

		float convY = 0.0f;
		for (int dy = -radiusY; dy <= radiusY; ++dy) {
			convY += tile[tileX + (tileY + dy) * tileWidth] * coeffsY[dy + radiusY];
		}

		// ensure you don't override tile on the next iteration while some warp is still reading data in y-pass
		// on last iteration not necessary
		if (p < radiusZ + outputsPerThread - 1) {
			__syncthreads();
		}

		for (int i = 0; i < outputsPerThread; ++i) {
			int relativeZ = p - i;
			if (relativeZ >= -radiusZ && relativeZ <= radiusZ) {
				results[i] += convY * coeffsZ[relativeZ + radiusZ];
			}
		}
	}

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y);
	const int z = baseZ;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		if (baseZ + i < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, baseZ + i, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = runSigmoid ? sigmoid(results[i], 0.0f, 10.0f) : results[i];
		}
	}
}

namespace cuda {
	namespace conv3d {
		void Conv3dNaive(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffs3d,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			const int coeffsSize = (filter_halfsize.x * 2 + 1) * (filter_halfsize.y * 2 + 1) * (filter_halfsize.z * 2 + 1);
			if (coeffs3d.m_size != size_t(coeffsSize)) {
				throw std::logic_error("CudaAlgoritms::Conv3d: invalid coeffs3d size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3d: input/output dim mismatch");
			}

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("Conv3dNaiveKernel", ctx, [&]() {
				Conv3dNaiveKernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					coeffs3d.m_ptr,
					filter_halfsize,
					dim
				);
			});
		}

		void Conv3dNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffs3d,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			const int coeffsSize = (filter_halfsize.x * 2 + 1) * (filter_halfsize.y * 2 + 1) * (filter_halfsize.z * 2 + 1);
			if (coeffs3d.m_size != size_t(coeffsSize)) {
				throw std::logic_error("CudaAlgoritms::Conv3d: invalid coeffs3d size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3d: input/output dim mismatch");
			}

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
			
			cuda::TimedCall("Conv3dNaiveSharedMemKernel", ctx, [&]() {
				Conv3dNaiveSharedMemKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					coeffs3d.m_ptr,
					filter_halfsize,
					dim
				);
			});
		}

		void Conv3dFusedSeparable(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (coeffsX.m_size != size_t(filter_halfsize.x * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparable: invalid coeffsX size");
			}

			if (coeffsY.m_size != size_t(filter_halfsize.y * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparable: invalid coeffsY size");
			}

			if (coeffsZ.m_size != size_t(filter_halfsize.z * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparable: invalid coeffsZ size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparable: input/output dim mismatch");
			}

			if (blockDim.z != 1u) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparable: blockDim.z must be equal to 1");
			}

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const size_t tileWidth = blockDim.x;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("Conv3dSeparableFusedKernel", ctx, [&]() {
				Conv3dSeparableFusedKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					coeffsX.m_ptr,
					coeffsY.m_ptr,
					coeffsZ.m_ptr,
					filter_halfsize,
					dim
				);
			});
		}

		void Conv3dFusedSeparableMultipleOutputs(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (coeffsX.m_size != size_t(filter_halfsize.x * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsX size");
			}

			if (coeffsY.m_size != size_t(filter_halfsize.y * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsY size");
			}

			if (coeffsZ.m_size != size_t(filter_halfsize.z * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsZ size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: input/output dim mismatch");
			}

			if (blockDim.z != 1u) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: blockDim.z must be equal to 1");
			}

			if (outputsPerThread < 2 || outputsPerThread > 13) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: unsupported outputsPerThread");
			}

			const auto blockDimForGrid = image::vec3ui{ blockDim.x, blockDim.y, unsigned int(outputsPerThread) };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimForGrid);

			const size_t tileWidth = blockDim.x;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 2) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<2>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<2, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<3>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<3, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<4>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<4, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<5>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<5, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<6>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<6, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<7>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<7, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<8>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<8, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<9>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<9, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<10>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<10, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 11) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<11>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<11, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 12) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<12>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<12, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
			else if (outputsPerThread == 13) {
				cuda::TimedCall("Conv3dSepFusedMultOutsKernel<13>", ctx, [&]() {
					Conv3dSeparableFusedMultipleOutKernel<13, 7, 5, 5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						dim
					);
				});
			}
		}

		void Conv3dFusedSeparableMultipleOutputs(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			int planesPerBatch,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (coeffsX.m_size != size_t(filter_halfsize.x * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsX size");
			}

			if (coeffsY.m_size != size_t(filter_halfsize.y * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsY size");
			}

			if (coeffsZ.m_size != size_t(filter_halfsize.z * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsZ size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: input/output dim mismatch");
			}

			if (blockDim.z != 1u) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: blockDim.z must be equal to 1");
			}

			if (outputsPerThread < 2 || outputsPerThread > 17) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: unsupported outputsPerThread");
			}

			if (planesPerBatch < 1 || planesPerBatch > outputsPerThread) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid planesPerBatch");
			}

			const auto blockDimForGrid = image::vec3ui{ blockDim.x, blockDim.y, unsigned int(outputsPerThread) };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimForGrid);

			const size_t tileWidth = blockDim.x;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * size_t(planesPerBatch) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 2) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<2>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<2, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<3>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<3, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<4>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<4, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<5>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<5, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<6>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<6, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<7>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<7, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<8>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<8, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<9>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<9, 7> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<10>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<10, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 11) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<11>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<11, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 12) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<12>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<12, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 13) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<13>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<13, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 14) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<14>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<14, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 15) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<15>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<15, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 16) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<16>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<16, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 17) {
				cuda::TimedCall("Conv3dSepFusedMultOutBatchKernel<17>", ctx, [&]() {
					Conv3dSeparableFusedMultOutBatchKernel<17, 2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						filter_halfsize,
						dim
					);
				});
			}
		}

		void Conv3dFusedSeparableMultipleOutputs(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> coeffsX,
			memory::GpuSpan<const float> coeffsY,
			memory::GpuSpan<const float> coeffsZ,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			bool runSigmoid,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (coeffsX.m_size != size_t(filter_halfsize.x * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsX size");
			}

			if (coeffsY.m_size != size_t(filter_halfsize.y * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsY size");
			}

			if (coeffsZ.m_size != size_t(filter_halfsize.z * 2 + 1)) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: invalid coeffsZ size");
			}

			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: input/output dim mismatch");
			}

			if (blockDim.z != 1u) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: blockDim.z must be equal to 1");
			}

			if (outputsPerThread < 2 || outputsPerThread > 13) {
				throw std::logic_error("CudaAlgoritms::Conv3dFusedSeparableMultipleOutputs: unsupported outputsPerThread");
			}

			const auto blockDimForGrid = image::vec3ui{ blockDim.x, blockDim.y, unsigned int(outputsPerThread) };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimForGrid);

			const size_t tileWidth = blockDim.x;
			const size_t tileHeight = blockDim.y + filter_halfsize.y * 2;

			const size_t sharedMemSize = tileWidth * tileHeight * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 9) {
				cuda::TimedCall("Conv3dSepFusedMultOutWithSigmKernel<9>", ctx, [&]() {
					Conv3dSepFusedMultOutWithSigmKernel<9, 7, 5, 3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						runSigmoid,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("Conv3dSepFusedMultOutWithSigmKernel<10>", ctx, [&]() {
					Conv3dSepFusedMultOutWithSigmKernel<10, 7, 5, 3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						runSigmoid,
						dim
					);
				});
			}
			else if (outputsPerThread == 11) {
				cuda::TimedCall("Conv3dSepFusedMultOutWithSigmKernel<11>", ctx, [&]() {
					Conv3dSepFusedMultOutWithSigmKernel<11, 7, 5, 3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						runSigmoid,
						dim
					);
				});
			}
			else if (outputsPerThread == 12) {
				cuda::TimedCall("Conv3dSepFusedMultOutWithSigmKernel<12>", ctx, [&]() {
					Conv3dSepFusedMultOutWithSigmKernel<12, 7, 5, 3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						runSigmoid,
						dim
					);
				});
			}
			else if (outputsPerThread == 13) {
				cuda::TimedCall("Conv3dSepFusedMultOutWithSigmKernel<13>", ctx, [&]() {
					Conv3dSepFusedMultOutWithSigmKernel<13, 7, 5, 3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						coeffsX.m_ptr,
						coeffsY.m_ptr,
						coeffsZ.m_ptr,
						runSigmoid,
						dim
					);
				});
			}
		}
	}
}
