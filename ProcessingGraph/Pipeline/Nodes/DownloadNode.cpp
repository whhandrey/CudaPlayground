#include "DownloadNode.h"
#include <Cuda/CudaCheck.h>
#include <Image/Image.h>

namespace {
	template<class Dst, class Src>
	struct IsDownloadCompatible
		: std::bool_constant<std::is_same_v<Dst, Src>>
	{
	};

	template<>
	struct IsDownloadCompatible<image::vec4uc, uchar4>
		: std::true_type
	{
	};

	// TODO: this must be moved out of cudaprocessingcore to common lib
	template <class DstSample, class SrcSample>
	void DownloadCompatible(image::GpuImageView<SrcSample> gpu_in, image::CpuImageView<DstSample> cpu_out, cudaStream_t stream) {

		static_assert(IsDownloadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for download");

		if (gpu_in.m_dim.x != cpu_out.m_dim.x || gpu_in.m_dim.y != cpu_out.m_dim.y) {
			throw std::logic_error("DownloadCompatible: trying to download an image with wrong dim");
		}

		const size_t rowBytes = gpu_in.m_dim.x * sizeof(SrcSample);
		if (gpu_in.m_pitch < rowBytes || cpu_out.m_pitch < rowBytes) {
			throw std::logic_error("DownloadCompatible: pitch is smaller than row size");
		}

		cudaCheck(cudaMemcpy2DAsync(
			cpu_out.m_ptr,
			cpu_out.m_dim.x * sizeof(DstSample),
			gpu_in.m_ptr,
			gpu_in.m_pitch,
			gpu_in.m_dim.x * sizeof(SrcSample),
			gpu_in.m_dim.y,
			cudaMemcpyDeviceToHost,
			stream
		));
	}
}

namespace pipeline {
	DownloadNode::DownloadNode(NodeId id, const DownloadNodeParams& params)
		: NodeBase(id, params.listener)
		, m_gpuInput(id, params.gpuInputName, params.registry, params.resRegistry, params.gpuInputDesc)
		, m_cpuOutput(id, params.cpuOutputName, params.registry, params.resRegistry, params.cpuOutputDesc)
	{
	}

	void DownloadNode::Execute(const CudaExecutionContext& ctx) {
		DownloadCompatible<image::vec4uc>(m_gpuInput.View(), m_cpuOutput.View(), ctx.stream);
	}
}
