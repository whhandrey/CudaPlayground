#include "BlockMatchingNode.h"

namespace pipeline {
	BlockMatchingNode::BlockMatchingNode(dataflow::NodeId id, dataflow::INodeChangeListener& listener)
		: NodeBase(id, listener)
		, m_params(*this, "blockMatchingParams")
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
