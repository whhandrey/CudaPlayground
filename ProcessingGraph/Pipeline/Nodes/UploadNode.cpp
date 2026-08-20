#include "UploadNode.h"
#include <Cuda/CudaCheck.h>
#include <Cuda/Image/GpuImageTransfer.h>

namespace pipeline {
	UploadNode::UploadNode(NodeId id, const UploadNodeParams& params)
		: NodeBase(id, params.listener)
		, m_cpuInput(id, params.cpuInputName, params.registry, params.resRegistry, params.cpuInputDesc)
		, m_gpuOutput(id, params.gpuOutputName, params.registry, params.resRegistry, params.gpuOutputDesc)
	{
	}

	void UploadNode::Execute(const CudaExecutionContext& ctx) {
		cuda::gpu_image::UploadCompatible(m_cpuInput.View(), m_gpuOutput.View(), ctx.stream);
	}
}
