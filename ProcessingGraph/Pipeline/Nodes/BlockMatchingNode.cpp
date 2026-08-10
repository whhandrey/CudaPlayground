#include "BlockMatchingNode.h"

namespace pipeline {
	BlockMatchingNode::BlockMatchingNode(NodeId id, dataflow::INodeChangeListener& listener, dataflow::IPortRegistry& registry)
		: NodeBase(id, listener)
		, m_params(id, "blockMatchingParams", registry, *this)
	{
	}

	void BlockMatchingNode::Execute(const CudaExecutionContext& ctx) {
		//cuda::KernelContext kernelCtx {
		//	ctx.stream,
		//	ctx.profiler
		//};

		//cuda::motion::BlockMatching(
		//	cuda::gpu_image::MakeImageView(m_prev),
		//	cuda::gpu_image::MakeImageView(m_curr),
		//	outView,
		//	m_params.Value(),
		//	kernelCtx
		//);
	}
}
