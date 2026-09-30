#include "UploadNode.h"
#include <Cuda/CudaCheck.h>
#include <Cuda/GpuImage/GpuImageTransfer.h>

namespace pipeline {
	UploadNode::UploadNode(NodeId id, const NodeBuildContext& ctx, const UploadNodeParams& params)
		: NodeBase(id, "GpuUploadNode", ctx.listener)
		, m_cpuInput(id, ctx.registry, ctx.resRegistry, params.cpuInputDesc)
		, m_gpuOutput(id, ctx.registry, ctx.resRegistry, params.gpuOutputDesc)
	{
	}

	void UploadNode::Execute(const CudaExecutionContext& ctx) {
		cuda::gpu_image::UploadCompatible(m_cpuInput.View(), m_gpuOutput.View(), ctx.stream);
	}
}
