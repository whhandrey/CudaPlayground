#include "DownloadNode.h"
#include <Cuda/CudaCheck.h>
#include <Cuda/Image/GpuImageTransfer.h>
#include <Image/Image.h>

namespace pipeline {
	DownloadNode::DownloadNode(NodeId id, const DownloadNodeParams& params)
		: NodeBase(id, params.listener)
		, m_gpuInput(id, params.gpuInputName, params.registry, params.resRegistry, params.gpuInputDesc)
		, m_cpuOutput(id, params.cpuOutputName, params.registry, params.resRegistry, params.cpuOutputDesc)
	{
	}

	void DownloadNode::Execute(const CudaExecutionContext& ctx) {
		cuda::gpu_image::DownloadCompatible<image::vec4uc>(m_gpuInput.View(), m_cpuOutput.View(), ctx.stream);
	}
}
