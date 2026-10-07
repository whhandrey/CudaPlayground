#include "LocalMinBlending.h"
#include "Common.cuh"
#include "../Math.cuh"
#include <vector>
#include <type_traits>

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

namespace cuda {
	namespace detail {
		bool CanRunFusedKernel(image::vec3ui blockDim, image::vec3ui blockOverlap) {
			return blockDim.x > blockOverlap.x
				&& blockDim.y > blockOverlap.y
				&& blockDim.z > blockOverlap.z;
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
			if (input.m_dim != output.m_dim) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHalo: input/output dim mismatch");
			}

			const image::vec3ui blockOverlap{ 0, unsigned int(filter_halfsize.y) * 2, unsigned int(filter_halfsize.z) * 2 };
			if (!detail::CanRunFusedKernel(blockDim, blockOverlap)) {
				throw std::logic_error("CudaAlgoritms::LocalMin3dFusedHalo: unsupported large filter_halfsize");
			}

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
			static_assert(std::is_trivially_copyable_v<BlendParams>, "BlendParams must be trivially copyable");
			static_assert(std::is_standard_layout_v<BlendParams>, "BlendParams must have standard layout");

			if (baseData.m_dim != gateField.m_dim || baseData.m_dim != activityField.m_dim || baseData.m_dim != filteredData.m_dim) {
				throw std::logic_error("CudaAlgoritms::MixingOp: input/output dim mismatch");
			}

			dim3 gridSize = cuda::math::DivUp(baseData.m_dim, blockDim);
			const int3 dim = {
				static_cast<int>(baseData.m_dim.x),
				static_cast<int>(baseData.m_dim.y),
				static_cast<int>(baseData.m_dim.z),
			};

			if (gateAlreadyProcessed) {
				cuda::TimedCall("BlendOpKernel<true>", ctx, [&]() {
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
			else {
				cuda::TimedCall("BlendOpKernel<false>", ctx, [&]() {
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
		}
	}
}
