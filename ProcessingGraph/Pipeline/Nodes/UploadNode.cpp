#include "UploadNode.h"
#include <Cuda/CudaCheck.h>

namespace {
	// TODO: this must be moved out of cudaprocessingcore to common lib
	template <class DstSample, class SrcSample>
	void UploadCompatible(
		const image::PinnedImageView<SrcSample>& in_cpu,
		image::GpuImageView<DstSample> out_gpu,
		cudaStream_t stream)
	{
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for upload");

		if (in_cpu.m_dim.x != out_gpu.m_dim.x || in_cpu.m_dim.y != out_gpu.m_dim.y) {
			throw std::logic_error("UploadCompatible: trying to upload an image with wrong dim");
		}

		cudaCheck(cudaMemcpy2DAsync(
			out_gpu.m_ptr,
			out_gpu.m_pitch,
			in_cpu.m_ptr,
			in_cpu.m_pitch,
			in_cpu.m_dim.x * sizeof(SrcSample),
			in_cpu.m_dim.y,
			cudaMemcpyHostToDevice,
			stream
		));
	}
}

namespace pipeline {
	UploadNode::UploadNode(NodeId id, const std::string& name, const UploadNodeParams& params)
		: NodeBase(id, params.listener)
		, m_cpuInput(id, name, params.registry, params.resRegistry, params.cpuInputDesc)
		, m_gpuOutput(id, name, params.registry, params.resRegistry, params.gpuOutputDesc)
	{
	}

	void UploadNode::Execute(const CudaExecutionContext& ctx) {
		UploadCompatible(m_cpuInput.View(), m_gpuOutput.View(), ctx.stream);
	}
}
