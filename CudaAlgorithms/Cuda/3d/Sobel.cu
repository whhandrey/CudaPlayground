#include "Sobel.h"
#include "../Math.cuh"
#include <vector>

#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__device__ int FlatIdx(int x, int y, int z, int width, int height) {
	return x + y * width + z * width * height;
}

__global__  void SobelMagnitude3dKernel(
	const float* __restrict__ input,
	float* output,
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

		tile[i] = input[FlatIdx(sampleX, sampleY, sampleZ, dim.x, dim.y)];
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
	output[FlatIdx(x, y, z, dim.x, dim.y)] = gradMag;
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
		void SobelMagnitude(
			const image::GpuVolumeView<float>& input,
			image::GpuVolumeView<float>& output,
			cuda::KernelContext& ctx)
		{



		}
	}
}
