#include "Conv3d.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <stdexcept>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

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

namespace cuda {
	namespace conv3d {
		void Conv3dNaiveSharedMem(
			image::GpuVolumeView<const float> input,
			memory::GpuSpan<const float> weights,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			const int coeffsSize = (filter_halfsize.x * 2 + 1) * (filter_halfsize.y * 2 + 1) * (filter_halfsize.z * 2 + 1);
			if (weights.m_size != size_t(coeffsSize)) {
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

			int3 dim = {
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
					weights.m_ptr,
					filter_halfsize,
					dim
				);
			});
		}
	}
}
