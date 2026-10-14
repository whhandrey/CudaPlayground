#include "LocalMinBlending.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <vector>
#include <type_traits>
#include <iostream>

#include <Cuda/GpuImage/GpuImageView.h>
#include <Cuda/TimedCudaCall.h>
#include <Cuda/MathUtils.h>

__global__  void LocalMin3dFusedHaloKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	// some y's will be pointing to the halo of particular blockDim
	// threadIdx.y:  0   1 | 2   3   4   5   6   7 | 8   9
	// global y :   -2  -1 | 0   1   2   3   4   5 | 6   7
	// role :         halo |     useful outputs    | halo
	int y = blockIdx.y * (blockDim.y - filter_halfsize.y * 2) + int(threadIdx.y) - filter_halfsize.y;
	int z = blockIdx.z * (blockDim.z - filter_halfsize.z * 2) + int(threadIdx.z) - filter_halfsize.z;

	// x-pass
	int y_idx = clamp(y, 0, dim.y - 1);
	int z_idx = clamp(z, 0, dim.z - 1);

	float sampleMin = FLT_MAX;
	for (int dx = -filter_halfsize.x; dx <=filter_halfsize.x ; ++dx) {
		int x_idx = clamp(x + dx, 0, dim.x - 1);

		const size_t offsetBytesToRow = OffsetToRowPitched(y_idx, z_idx, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		sampleMin = min(sampleMin, row[x_idx]);
	}

	const int tileIdx = FlatIdx(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y);
	tile[tileIdx] = sampleMin;

	__syncthreads();

	const int threadIdxY = threadIdx.y - filter_halfsize.y;
	const int blockDimY = blockDim.y - filter_halfsize.y * 2;

	float* xyTile = tile + (blockDim.x * blockDim.y * blockDim.z);
	const bool processY = threadIdx.y >= filter_halfsize.y && threadIdx.y < blockDim.y - filter_halfsize.y;

	if (processY) {
		sampleMin = FLT_MAX;

		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			const int tileXIdx = FlatIdx(threadIdx.x, threadIdx.y + dy, threadIdx.z, blockDim.x, blockDim.y);
			sampleMin = min(sampleMin, tile[tileXIdx]);
		}

		const int tileXYIdx = FlatIdx(threadIdx.x, threadIdxY, threadIdx.z, blockDim.x, blockDimY);
		xyTile[tileXYIdx] = sampleMin;
	}

	__syncthreads();

	const bool processZ = processY
		&& threadIdx.z >= filter_halfsize.z && threadIdx.z < blockDim.z - filter_halfsize.z
		&& x < dim.x && y < dim.y && z < dim.z;

	if (processZ) {
		sampleMin = FLT_MAX;

		for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; ++dz) {
			const int tileXYIdx = FlatIdx(threadIdx.x, threadIdxY, threadIdx.z + dz, blockDim.x, blockDimY);
			sampleMin = min(sampleMin, xyTile[tileXYIdx]);
		}

		const size_t offsetBytes = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
		float* row = (float*)((char*)output + offsetBytes);

		row[x] = sampleMin;
	}
}

__global__ void BlendOpKernel(
	image::GpuVolumeView<float> baseData,
	image::GpuVolumeView<const float> gateField,
	image::GpuVolumeView<const float> activityField,
	image::GpuVolumeView<const float> filteredData,
	bool gateAlreadyProcessed,
	cuda::erode3d::BlendParams params,
	int3 dim)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	float gate = ReadPitched(gateField, x, y, z);
	const float activity = ReadPitched(activityField, x, y, z);

	if (!gateAlreadyProcessed) {
		gate = expf(-params.gateSlope * fabsf(gate - params.gateThreshold));
	}

	float blend = __saturatef(0.5f * gate + 0.5f * activity);

	const float blendExponent = gateAlreadyProcessed ? params.alternateBlendWeight : params.defaultBlendWeight;
	blend = powf(blend, blendExponent);

	float filtered = ReadPitched(filteredData, x, y, z) * params.auxiliaryScale;

	const float original = ReadPitched(baseData, x, y, z);
	const float mixed = original + blend * (filtered - original);

	const float result = fminf(original, mixed);
	WritePitched(baseData, result, x, y, z);
}

__global__  void LocalMin3dFusedSeparableKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

	const int tileXWidth = blockDim.x;
	const int tileXHeight = blockDim.y + filter_halfsize.y * 2;
	const int tileXDepth = blockDim.z + filter_halfsize.z * 2;

	const int tileXSize = tileXWidth * tileXHeight * tileXDepth;
	const int tileXPlaneSize = tileXWidth * tileXHeight;

	const int blockSize = blockDim.x * blockDim.y * blockDim.z;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z * blockDim.z;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

	// cooperative localMinX pass into sharedMem
	for (int i = tid; i < tileXSize; i += blockSize) {
		const int tileZ = i / tileXPlaneSize;
		const int tileXYIdx = i % tileXPlaneSize;

		const int tileY = tileXYIdx / tileXWidth;
		const int tileX = tileXYIdx % tileXWidth;

		int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);
		int z = clamp(originZ + tileZ - filter_halfsize.z, 0, dim.z - 1);

		float sampleMin = FLT_MAX;
		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
			int x = clamp(originX + tileX + dx, 0, dim.x - 1);
			sampleMin = fminf(sampleMin, row[x]);
		}

		tile[i] = sampleMin;
	}

	__syncthreads();

	float* tileXY = tile + tileXSize;

	const int tileXYWidth = blockDim.x;
	const int tileXYHeight = blockDim.y;
	const int tileXYDepth = blockDim.z + filter_halfsize.z * 2;
	const int tileXYSize = tileXYWidth * tileXYHeight * tileXYDepth;
	const int tileXYPlaneSize = tileXYWidth * tileXYHeight;

	// cooperative localMinXY pass into sharedMem
	for (int i = tid; i < tileXYSize; i += blockSize) {
		const int tileZ = i / tileXYPlaneSize;
		const int tileXYIdx = i % tileXYPlaneSize;

		const int tileY = tileXYIdx / tileXYWidth;
		const int tileX = tileXYIdx % tileXYWidth;
		
		// this is basically center in the tileX
		// since it has halo we need to shift it to filter_halfsize.y
		const int centerY = tileY + filter_halfsize.y;

		float sampleMin = FLT_MAX;
		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			const int flatIdx = tileX + (centerY + dy) * tileXWidth + tileZ * tileXPlaneSize;
			sampleMin = fminf(sampleMin, tile[flatIdx]);
		}

		tileXY[i] = sampleMin;
	}

	__syncthreads();

	float sampleMin = FLT_MAX;
	const int centerZ = threadIdx.z + filter_halfsize.z;

	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; ++dz) {
		const int flatIdx = threadIdx.x + threadIdx.y * tileXYWidth + (centerZ + dz) * tileXYPlaneSize;
		sampleMin = fminf(sampleMin, tileXY[flatIdx]);
	}

	const int x = originX + threadIdx.x;
	const int y = originY + threadIdx.y;
	const int z = originZ + threadIdx.z;

	if (x >= dim.x || y >= dim.y || z >= dim.z) {
		return;
	}

	const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
	float* row = (float*)((char*)output + offsetBytesToRow);

	row[x] = sampleMin;
}

template <int planesPerBatch>
__global__  void LocalMin3dFusedHaloStreamZKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float tile[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	// some y's will be pointing to the halo of particular blockDim
	// threadIdx.y:  0   1 | 2   3   4   5   6   7 | 8   9
	// global y :   -2  -1 | 0   1   2   3   4   5 | 6   7
	// role :         halo |     useful outputs    | halo
	int y = blockIdx.y * (blockDim.y - filter_halfsize.y * 2) + int(threadIdx.y) - filter_halfsize.y;
	int z = blockIdx.z;

	const int y_idx = clamp(y, 0, dim.y - 1);
	const bool processY = threadIdx.y >= filter_halfsize.y && threadIdx.y < blockDim.y - filter_halfsize.y;

	float sampleMinXYZ = FLT_MAX;
	for (int dz = -filter_halfsize.z; dz <= filter_halfsize.z; dz += planesPerBatch) {
		const int validPlanes = min(planesPerBatch, filter_halfsize.z - dz + 1);

		for (int planeIndex = 0; planeIndex < validPlanes; ++planeIndex) {

			float sampleMin = FLT_MAX;
			const int z_idx = clamp(z + dz + planeIndex, 0, dim.z - 1);

			const size_t offsetBytesToRow = OffsetToRowPitched(y_idx, z_idx, inPitch, inSlicePitch);
			const float* row = (const float*)((const char*)input + offsetBytesToRow);

			for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
				int x_idx = clamp(x + dx, 0, dim.x - 1);
				sampleMin = fminf(sampleMin, row[x_idx]);
			}

			const int tileIdx = FlatIdx(threadIdx.x, threadIdx.y, planeIndex, blockDim.x, blockDim.y);
			tile[tileIdx] = sampleMin;
		}

		__syncthreads();

		if (processY) {
			float sampleMinXY = FLT_MAX;

			for (int planeIndex = 0; planeIndex < validPlanes; ++planeIndex) {

				float sampleMin = FLT_MAX;

				for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
					const int flatIdx = threadIdx.x + (threadIdx.y + dy) * blockDim.x + planeIndex * (blockDim.x * blockDim.y);
					sampleMin = fminf(sampleMin, tile[flatIdx]);
				}

				sampleMinXY = fminf(sampleMinXY, sampleMin);
			}
			
			sampleMinXYZ = fminf(sampleMinXYZ, sampleMinXY);
		}

		if (dz + planesPerBatch <= filter_halfsize.z) {
			__syncthreads();
		}
	}

	if (processY && x < dim.x && y < dim.y && z < dim.z) {
		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, outPitch, outSlicePitch);
		float* row = (float*)((char*)output + offsetBytesToRow);

		row[x] = sampleMinXYZ;
	}
}

template <int outputsPerThread>
__global__  void LocalMin3dSepFusedMultipleOutputsKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float rawTile[];

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

		rawTile[i] = row[x];
	}

	__syncthreads();

	float* xTile = rawTile + tileSize;
	const int xTileWidth = blockDim.x;

	const int xTileSize = xTileWidth * tileHeight * tileDepth;
	const int xTilePlaneSize = xTileWidth * tileHeight;

	for (int i = tid; i < xTileSize; i += blockSize) {
		const int tileZ = i / xTilePlaneSize;
		const int tileXY = i % xTilePlaneSize;

		const int tileY = tileXY / xTileWidth;
		const int rawCenterTileX = (tileXY % xTileWidth) + filter_halfsize.x;

		float sampleMin = FLT_MAX;
		for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
			const int flatIdx = FlatIdx(rawCenterTileX + dx, tileY, tileZ, tileWidth, tileHeight);
			sampleMin = fminf(sampleMin, rawTile[flatIdx]);
		}

		xTile[i] = sampleMin;
	}

	__syncthreads();

	const int x = originX + threadIdx.x;
	const int y = originY + threadIdx.y;
	const int firstOutputZ = originZ + threadIdx.z * outputsPerThread;

	if (x >= dim.x || y >= dim.y || firstOutputZ >= dim.z) {
		return;
	}

	float zMin[outputsPerThread];

	#pragma unroll
	for (int i = 0; i < outputsPerThread; ++i) {
		zMin[i] = FLT_MAX;
	}

	const int centerY = threadIdx.y + filter_halfsize.y;
	const int groupFirstTileZ = threadIdx.z * outputsPerThread;

	for (int dz = -filter_halfsize.z; dz < filter_halfsize.z + outputsPerThread; ++dz) {

		float sampleMin = FLT_MAX;
		const int tileZ = groupFirstTileZ + dz + filter_halfsize.z;

		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			const int flatIdx = FlatIdx(threadIdx.x, centerY + dy, tileZ, xTileWidth, tileHeight);
			sampleMin = fminf(sampleMin, xTile[flatIdx]);
		}

		for (int j = 0; j < outputsPerThread; j++) {
			const int relativeZ = j - dz;
			if (relativeZ >= -filter_halfsize.z && relativeZ <= filter_halfsize.z) {
				zMin[j] = fminf(zMin[j], sampleMin);
			}
		}
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		const int outputZ = firstOutputZ + i;

		if (outputZ < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, outputZ, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = zMin[i];
		}
	}
}

template <int outputsPerThread>
__global__  void LocalMin3dSepFusedMultipleOutputsV2Kernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float xTile[];

	const int outputDepth = blockDim.z * outputsPerThread;

	const int tileWidth = blockDim.x;
	const int tileHeight = blockDim.y + filter_halfsize.y * 2;
	const int tileDepth = outputDepth + filter_halfsize.z * 2;

	const int tileSize = tileWidth * tileHeight * tileDepth;
	const int tilePlaneSize = tileWidth * tileHeight;

	const int blockSize = blockDim.x * blockDim.y * blockDim.z;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * blockDim.y;
	const int originZ = blockIdx.z * outputDepth;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

	for (int i = tid; i < tileSize; i += blockSize) {
		const int tileZ = i / tilePlaneSize;
		const int tileXY = i % tilePlaneSize;

		const int tileY = tileXY / tileWidth;
		const int tileX = tileXY % tileWidth;

		const int y = clamp(originY + tileY - filter_halfsize.y, 0, dim.y - 1);
		const int z = clamp(originZ + tileZ - filter_halfsize.z, 0, dim.z - 1);

		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		float sampleMin = FLT_MAX;
		for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
			const int x = clamp(originX + tileX + dx, 0, dim.x - 1);
			sampleMin = fminf(sampleMin, row[x]);
		}

		xTile[i] = sampleMin;
	}

	__syncthreads();

	const int x = originX + threadIdx.x;
	const int y = originY + threadIdx.y;
	const int firstOutputZ = originZ + threadIdx.z * outputsPerThread;

	if (x >= dim.x || y >= dim.y || firstOutputZ >= dim.z) {
		return;
	}

	float zMin[outputsPerThread];

	#pragma unroll
	for (int i = 0; i < outputsPerThread; ++i) {
		zMin[i] = FLT_MAX;
	}

	const int centerY = threadIdx.y + filter_halfsize.y;
	const int groupFirstTileZ = threadIdx.z * outputsPerThread;

	for (int dz = -filter_halfsize.z; dz < filter_halfsize.z + outputsPerThread; ++dz) {

		float sampleMin = FLT_MAX;
		const int tileZ = groupFirstTileZ + dz + filter_halfsize.z;

		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			const int flatIdx = FlatIdx(threadIdx.x, centerY + dy, tileZ, tileWidth, tileHeight);
			sampleMin = fminf(sampleMin, xTile[flatIdx]);
		}

		for (int j = 0; j < outputsPerThread; j++) {
			const int relativeZ = j - dz;
			if (relativeZ >= -filter_halfsize.z && relativeZ <= filter_halfsize.z) {
				zMin[j] = fminf(zMin[j], sampleMin);
			}
		}
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		const int outputZ = firstOutputZ + i;

		if (outputZ < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, outputZ, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = zMin[i];
		}
	}
}

template <int outputsPerThread>
__global__  void LocalMin3dSepHaloYMultipleOutputsKernel(
	const float* __restrict__ input,
	size_t inPitch,
	size_t inSlicePitch,
	float* __restrict__ output,
	size_t outPitch,
	size_t outSlicePitch,
	image::vec3i filter_halfsize,
	int3 dim)
{
	extern __shared__ float xTile[];

	const int outputDepth = blockDim.z * outputsPerThread;
	const int tileDepth = outputDepth + filter_halfsize.z * 2;

	const int originX = blockIdx.x * blockDim.x;
	const int originY = blockIdx.y * (blockDim.y - filter_halfsize.y * 2);
	const int originZ = blockIdx.z * outputDepth;

	for (int tileZ = threadIdx.z; tileZ < tileDepth; tileZ += blockDim.z) {
		int y = clamp(originY + int(threadIdx.y) - filter_halfsize.y, 0, dim.y - 1);
		int z = clamp(originZ + tileZ - filter_halfsize.z, 0, dim.z - 1);

		const size_t offsetBytesToRow = OffsetToRowPitched(y, z, inPitch, inSlicePitch);
		const float* row = (const float*)((const char*)input + offsetBytesToRow);

		float sampleMin = FLT_MAX;
		for (int dx = -filter_halfsize.x; dx <= filter_halfsize.x; ++dx) {
			const int x = clamp(originX + int(threadIdx.x) + dx, 0, dim.x - 1);
			sampleMin = fminf(sampleMin, row[x]);
		}

		const int flatIdx = threadIdx.x + threadIdx.y * blockDim.x + tileZ * (blockDim.x * blockDim.y);
		xTile[flatIdx] = sampleMin;
	}

	__syncthreads();

	const int x = originX + int(threadIdx.x);
	const int y = originY + int(threadIdx.y) - filter_halfsize.y;

	const int firstOutputZ = originZ + int(threadIdx.z) * outputsPerThread;
	const bool processY = threadIdx.y >= filter_halfsize.y && threadIdx.y < blockDim.y - filter_halfsize.y;

	if (x >= dim.x || y >= dim.y || firstOutputZ >= dim.z || !processY) {
		return;
	}

	float zMin[outputsPerThread];

	#pragma unroll
	for (int i = 0; i < outputsPerThread; ++i) {
		zMin[i] = FLT_MAX;
	}
	
	const int groupFirstTileZ = threadIdx.z * outputsPerThread;
	for (int dz = -filter_halfsize.z; dz < filter_halfsize.z + outputsPerThread; ++dz) {

		float sampleMin = FLT_MAX;
		const int tileZ = groupFirstTileZ + dz + filter_halfsize.z;

		for (int dy = -filter_halfsize.y; dy <= filter_halfsize.y; ++dy) {
			const int flatIdx = FlatIdx(threadIdx.x, threadIdx.y + dy, tileZ, blockDim.x, blockDim.y);
			sampleMin = fminf(sampleMin, xTile[flatIdx]);
		}

		for (int j = 0; j < outputsPerThread; j++) {
			const int relativeZ = j - dz;
			if (relativeZ >= -filter_halfsize.z && relativeZ <= filter_halfsize.z) {
				zMin[j] = fminf(zMin[j], sampleMin);
			}
		}
	}

	for (int i = 0; i < outputsPerThread; ++i) {
		const int outputZ = firstOutputZ + i;

		if (outputZ < dim.z) {
			const size_t offsetBytesToRow = OffsetToRowPitched(y, outputZ, outPitch, outSlicePitch);
			float* row = (float*)((char*)output + offsetBytesToRow);

			row[x] = zMin[i];
		}
	}
}

namespace cuda {
	namespace detail {
		bool CanRunFusedKernel(image::vec3ui blockDim, image::vec3ui blockOverlap) {
			return blockDim.x > blockOverlap.x
				&& blockDim.y > blockOverlap.y
				&& blockDim.z > blockOverlap.z;
		}

		void CheckValidLocalMin3dInput(
			image::vec3ui inputDim,
			image::vec3ui outputDim,
			image::vec3ui blockDim,
			image::vec3i filter_halfsize)
		{
			if (inputDim != outputDim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHalo: input/output dim mismatch");
			}

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, unsigned int(filter_halfsize.z) * 2 };
			if (!CanRunFusedKernel(blockDim, blockOverlap)) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHalo: unsupported large filter_halfsize");
			}
		}

		void CheckValidBlend3dInput(
			image::vec3ui baseDataDim,
			image::vec3ui gateFieldDim,
			image::vec3ui activityFieldDim,
			image::vec3ui filteredDataDim)
		{
			using cuda::erode3d::BlendParams;

			static_assert(std::is_trivially_copyable_v<BlendParams>, "BlendParams must be trivially copyable");
			static_assert(std::is_standard_layout_v<BlendParams>, "BlendParams must have standard layout");

			if (baseDataDim != gateFieldDim || baseDataDim != activityFieldDim || baseDataDim != filteredDataDim) {
				throw std::logic_error("CudaAlgoritms::Blend3dOp: input/output dim mismatch");
			}
		}
	}

	namespace erode3d {
		void LocalMin3dFusedHalo(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			detail::CheckValidLocalMin3dInput(input.m_dim, output.m_dim, blockDim, filter_halfsize);

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, unsigned int(filter_halfsize.z) * 2 };
			const auto blockDimGrid = image::vec3ui{ blockDim.x, blockDim.y - blockOverlap.y, blockDim.z - blockOverlap.z };

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);

			const size_t tileSizeX = blockDim.x * blockDim.y * blockDim.z;
			const size_t tileSizeXY = blockDim.x * (blockDim.y - filter_halfsize.y * 2) * blockDim.z;

			const size_t sharedMemSize = (tileSizeX + tileSizeXY) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("LocalMin3dFusedHaloKernel", ctx, [&]() {
				LocalMin3dFusedHaloKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					filter_halfsize,
					dim
				);
			});
		}

		void LocalMin3dFusedSeparable(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedSeparable: input/output dim mismatch");
			}

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDim);

			const size_t tileSizeX = blockDim.x * (blockDim.y + filter_halfsize.y * 2) * (blockDim.z + filter_halfsize.z * 2);
			const size_t tileSizeXY = blockDim.x * blockDim.y * (blockDim.z + filter_halfsize.z * 2);

			const size_t sharedMemSize = (tileSizeX + tileSizeXY) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			cuda::TimedCall("LocalMin3dFusedSeparableKernel", ctx, [&]() {
				LocalMin3dFusedSeparableKernel <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
					input.m_ptr,
					input.m_pitch,
					input.m_slicePitch,
					output.m_ptr,
					output.m_pitch,
					output.m_slicePitch,
					filter_halfsize,
					dim
				);
			});
		}

		void LocalMin3dFusedHaloStreamZ(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int planesPerBatch,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloStreamZ: input/output dim mismatch");
			}

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, 0 };
			if (!detail::CanRunFusedKernel(blockDim, blockOverlap)) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloStreamZ: unsupported large filter_halfsize.y");
			}

			if (planesPerBatch > filter_halfsize.z * 2 + 1) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloStreamZ: requested more planesPerBatch than filterLengthZ");
			}

			if (blockDim.z != 1u) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloStreamZ: blockDim.z must be 1");
			}

			const auto blockDimGrid = image::vec3ui{ blockDim.x, blockDim.y - unsigned int(filter_halfsize.y) * 2, 1u };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);

			const size_t sharedMemSize = (blockDim.x * blockDim.y * planesPerBatch) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (planesPerBatch == 1) {
				cuda::TimedCall("LocalMin3dFusedHaloStreamZKernel<1>", ctx, [&]() {
					LocalMin3dFusedHaloStreamZKernel<1> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (planesPerBatch == 2) {
				cuda::TimedCall("LocalMin3dFusedHaloStreamZKernel<2>", ctx, [&]() {
					LocalMin3dFusedHaloStreamZKernel<2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (planesPerBatch == 3) {
				cuda::TimedCall("LocalMin3dFusedHaloStreamZKernel<3>", ctx, [&]() {
					LocalMin3dFusedHaloStreamZKernel<3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (planesPerBatch == 4) {
				cuda::TimedCall("LocalMin3dFusedHaloStreamZKernel<4>", ctx, [&]() {
					LocalMin3dFusedHaloStreamZKernel<4> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (planesPerBatch == 5) {
				cuda::TimedCall("LocalMin3dFusedHaloStreamZKernel<5>", ctx, [&]() {
					LocalMin3dFusedHaloStreamZKernel<5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
		}

		template <int outputsPerThread>
		void PrintLocalMinOccupancy(image::vec3ui blockDim, image::vec3i filter_halfsize) {
			const unsigned int outputDepth = blockDim.z * static_cast<unsigned int>(outputsPerThread);

			const size_t rawTileCount =
				(blockDim.x + 2 * filter_halfsize.x) *
				(blockDim.y + 2 * filter_halfsize.y) *
				(outputDepth + 2 * filter_halfsize.z);

			const size_t xTileCount =
				blockDim.x *
				(blockDim.y + 2 * filter_halfsize.y) *
				(outputDepth + 2 * filter_halfsize.z);

			const size_t sharedMemSize = (rawTileCount + xTileCount) * sizeof(float);
			const int threadsPerBlock = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

			int device = 0;
			cudaGetDevice(&device);

			cudaDeviceProp prop{};
			cudaGetDeviceProperties(&prop, device);

			int activeBlocksPerSM = 0;
			
			cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&activeBlocksPerSM,
				LocalMin3dSepFusedMultipleOutputsKernel<outputsPerThread>,
				threadsPerBlock,
				sharedMemSize)
			);

			const int warpsPerBlock = (threadsPerBlock + prop.warpSize - 1) / prop.warpSize;
			const int activeWarpsPerSM = activeBlocksPerSM * warpsPerBlock;
			const int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;

			const float occupancy = static_cast<float>(activeWarpsPerSM) / static_cast<float>(maxWarpsPerSM);

			std::cout
				<< "Kernel<" << outputsPerThread << ">\n"
				<< "Block: "
				<< blockDim.x << "x"
				<< blockDim.y << "x"
				<< blockDim.z << "\n"
				<< "Threads/block: " << threadsPerBlock << "\n"
				<< "Shared/block: " << sharedMemSize << " bytes\n"
				<< "Active blocks/SM: " << activeBlocksPerSM << "\n"
				<< "Active warps/SM: " << activeWarpsPerSM
				<< " / " << maxWarpsPerSM << "\n"
				<< "Theoretical occupancy: "
				<< occupancy * 100.0f << "%\n";
		}

		void LocalMin3dSepFusedMultipleOutputs(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dSepFusedMultipleOutputs: input/output dim mismatch");
			}

			if (outputsPerThread < 1 || outputsPerThread > 10) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dSepFusedMultipleOutputs: invalid outputsPerThread");
			}

			//static bool print = true;
			//if (print) {
			//	PrintLocalMinOccupancy<6>({ 16, 8, 2 }, { 1, 1, 1 });
			//	print = false;
			//}

			//return;

			const unsigned int depthZ = blockDim.z * unsigned int(outputsPerThread);

			const auto blockDimGrid = image::vec3ui{ blockDim.x, blockDim.y, depthZ };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);

			const size_t rawTileCount = (blockDim.x + filter_halfsize.x * 2)
				* (blockDim.y + filter_halfsize.y * 2)
				* (depthZ + filter_halfsize.z * 2);

			const size_t xTileCount = blockDim.x
				* (blockDim.y + filter_halfsize.y * 2)
				* (depthZ + filter_halfsize.z * 2);

			const size_t sharedMemSize = (rawTileCount + xTileCount) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 1) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<1>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<1> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 2) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<2>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<3>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<4>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<4> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<5>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
						);
					});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<6>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<6> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<7>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<7> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<8>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<8> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<9>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<9> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernel<10>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsKernel<10> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
		}

		void LocalMin3dSepFusedMultipleOutputsV2(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dSepFusedMultipleOutputsV2: input/output dim mismatch");
			}

			if (outputsPerThread < 1 || outputsPerThread > 10) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dSepFusedMultipleOutputsV2: invalid outputsPerThread");
			}

			const unsigned int depthZ = blockDim.z * unsigned int(outputsPerThread);

			const auto blockDimGrid = image::vec3ui{ blockDim.x, blockDim.y, depthZ };
			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);

			const size_t xTileCount = blockDim.x
				* (blockDim.y + filter_halfsize.y * 2)
				* (depthZ + filter_halfsize.z * 2);

			const size_t sharedMemSize = xTileCount * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 1) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<1>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<1> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 2) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<2>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<3>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<4>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<4> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<5>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<6>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<6> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<7>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<7> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<8>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<8> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<9>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<9> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("LocalMin3dSepFusedMultOutsKernelV2<10>", ctx, [&]() {
					LocalMin3dSepFusedMultipleOutputsV2Kernel<10> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
		}

		void LocalMin3dFusedHaloYMultipleOutputs(
			image::GpuVolumeView<const float> input,
			image::GpuVolumeView<float> output,
			image::vec3i filter_halfsize,
			int outputsPerThread,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloYMultipleOutputs: input/output dim mismatch");
			}

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, 0 };
			if (!detail::CanRunFusedKernel(blockDim, blockOverlap)) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloYMultipleOutputs: unsupported large filter_halfsize.y");
			}

			if (outputsPerThread < 1 || outputsPerThread > 10) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHaloYMultipleOutputs: unsupported or invalid outputsPerThread");
			}

			const unsigned int depthZ = blockDim.z * unsigned int(outputsPerThread);
			const auto blockDimGrid = image::vec3ui {
				blockDim.x,
				blockDim.y - unsigned int(filter_halfsize.y) * 2,
				depthZ
			};

			dim3 gridSize = cuda::math::DivUp(input.m_dim, blockDimGrid);
			const size_t sharedMemSize = blockDim.x * blockDim.y * (depthZ + unsigned int(filter_halfsize.z) * 2) * sizeof(float);

			const int3 dim = {
				static_cast<int>(input.m_dim.x),
				static_cast<int>(input.m_dim.y),
				static_cast<int>(input.m_dim.z),
			};

			if (outputsPerThread == 1) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<1>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<1> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 2) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<2>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<2> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 3) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<3>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<3> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 4) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<4>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<4> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 5) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<5>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<5> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 6) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<6>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<6> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 7) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<7>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<7> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 8) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<8>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<8> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 9) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<9>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<9> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
			else if (outputsPerThread == 10) {
				cuda::TimedCall("LocalMin3dSepHaloYMultOutsKernel<10>", ctx, [&]() {
					LocalMin3dSepHaloYMultipleOutputsKernel<10> <<<gridSize, cuda::math::vecTodim3(blockDim), sharedMemSize, ctx.stream>>> (
						input.m_ptr,
						input.m_pitch,
						input.m_slicePitch,
						output.m_ptr,
						output.m_pitch,
						output.m_slicePitch,
						filter_halfsize,
						dim
					);
				});
			}
		}

		void BlendOp(
			image::GpuVolumeView<float> baseData,
			image::GpuVolumeView<const float> gateField,
			image::GpuVolumeView<const float> activityField,
			image::GpuVolumeView<const float> filteredData,
			bool gateAlreadyProcessed,
			const BlendParams& params,
			cuda::KernelContext ctx,
			image::vec3ui blockDim)
		{
			detail::CheckValidBlend3dInput(baseData.m_dim, gateField.m_dim, activityField.m_dim, filteredData.m_dim);

			dim3 gridSize = cuda::math::DivUp(baseData.m_dim, blockDim);
			const int3 dim = {
				static_cast<int>(baseData.m_dim.x),
				static_cast<int>(baseData.m_dim.y),
				static_cast<int>(baseData.m_dim.z),
			};

			const std::string kernelName = gateAlreadyProcessed ? "BlendOpKernel(true)" : "BlendOpKernel(false)";
			cuda::TimedCall(kernelName, ctx, [&]() {
				BlendOpKernel <<<gridSize, cuda::math::vecTodim3(blockDim), 0, ctx.stream>>> (
					baseData,
					gateField,
					activityField,
					filteredData,
					gateAlreadyProcessed,
					params,
					dim
				);
			});
		}

		void LocalMin3dThenBlendOp(
			image::GpuVolumeView<float> baseData,
			image::GpuVolumeView<const float> gateField,
			image::GpuVolumeView<const float> activityField,
			image::GpuVolumeView<float> filteredData,
			bool gateAlreadyProcessed,
			const BlendParams& params,
			image::vec3i filter_halfsize,
			cuda::KernelContext ctx,
			image::vec3ui blockDimLocMin,
			image::vec3ui blockDimBlend)
		{
			detail::CheckValidLocalMin3dInput(baseData.m_dim, filteredData.m_dim, blockDimLocMin, filter_halfsize);
			detail::CheckValidBlend3dInput(baseData.m_dim, gateField.m_dim, activityField.m_dim, filteredData.m_dim);

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, unsigned int(filter_halfsize.z) * 2 };
			const auto blockDimGridLocMin = image::vec3ui{ blockDimLocMin.x, blockDimLocMin.y - blockOverlap.y, blockDimLocMin.z - blockOverlap.z };

			dim3 gridSizeLocMin = cuda::math::DivUp(baseData.m_dim, blockDimGridLocMin);
			dim3 gridSizeBlend = cuda::math::DivUp(baseData.m_dim, blockDimBlend);

			const size_t tileSizeX = blockDimLocMin.x * blockDimLocMin.y * blockDimLocMin.z;
			const size_t tileSizeXY = blockDimLocMin.x * (blockDimLocMin.y - filter_halfsize.y * 2) * blockDimLocMin.z;

			const size_t sharedMemSize = (tileSizeX + tileSizeXY) * sizeof(float);

			const int3 dim = {
				static_cast<int>(baseData.m_dim.x),
				static_cast<int>(baseData.m_dim.y),
				static_cast<int>(baseData.m_dim.z),
			};

			cuda::TimedCall("LocalMin3dAndBlend3dKernels", ctx, [&]() {
				LocalMin3dFusedHaloKernel <<<gridSizeLocMin, cuda::math::vecTodim3(blockDimLocMin), sharedMemSize, ctx.stream>>> (
					baseData.m_ptr,
					baseData.m_pitch,
					baseData.m_slicePitch,
					filteredData.m_ptr,
					filteredData.m_pitch,
					filteredData.m_slicePitch,
					filter_halfsize,
					dim
				);

				cudaCheck(cudaPeekAtLastError());

				BlendOpKernel <<<gridSizeBlend, cuda::math::vecTodim3(blockDimBlend), 0, ctx.stream>>> (
					baseData,
					gateField,
					activityField,
					cuda::gpu_image::ToConstView(filteredData),
					gateAlreadyProcessed,
					params,
					dim
				);

				cudaCheck(cudaPeekAtLastError());
			});
		}
	}
}
