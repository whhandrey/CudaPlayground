#include "DownloadNode.h"
#include <Cuda/CudaCheck.h>
#include <Cuda/Image/GpuImageTransfer.h>
#include <Image/Image.h>

namespace pipeline {
	DownloadNode::DownloadNode(NodeId id, const NodeBuildContext& ctx, const DownloadNodeParams& params)
		: NodeBase(id, "DownloadNode", ctx.listener)
		, m_gpuInput(id, ctx.registry, ctx.resRegistry, params.gpuInputDesc)
		, m_cpuOutput(id, ctx.registry, ctx.resRegistry, params.cpuOutputDesc)
	{
	}

	void DownloadNode::Execute(const CudaExecutionContext& ctx) {
		cuda::gpu_image::DownloadCompatible<image::vec4uc>(m_gpuInput.View(), m_cpuOutput.View(), ctx.stream);
	}
}
