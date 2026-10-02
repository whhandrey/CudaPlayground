#include "Sobel.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <vector>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__  void SobelMagnitude3dKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	const float* sobelX,
	const float* sobelY,
	const float* sobelZ,
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

namespace {
	struct SobelKernels {
		std::vector<float> sobelX;
		std::vector<float> sobelY;
		std::vector<float> sobelZ;
	};

	SobelKernels SobelKernels3d() {
		constexpr float smooth[] = { 0.25f, 0.5f, 0.25f };
		constexpr float derivative[] = { -0.5f, 0.0f, 0.5f };
		const int sobelSize = 3;

		std::vector<float> sobelX;
		std::vector<float> sobelY;
		std::vector<float> sobelZ;

		for (int z = 0; z < sobelSize; ++z) {
			for (int y = 0; y < sobelSize; ++y) {
				for (int x = 0; x < sobelSize; ++x) {
					sobelX.push_back(derivative[x] * smooth[y] * smooth[z]);
					sobelY.push_back(smooth[x] * derivative[y] * smooth[z]);
					sobelZ.push_back(smooth[x] * smooth[y] * derivative[z]);
				}
			}
		}

		return { sobelX, sobelY, sobelZ };
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

			cuda::TimedCall("SobelMagnitude3dKernel", ctx, [&]() {
				SobelMagnitude3dKernel << <gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream >> > (
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
	}
}
