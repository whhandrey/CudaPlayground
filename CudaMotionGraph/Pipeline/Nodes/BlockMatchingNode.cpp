#include "BlockMatchingNode.h"

namespace pipeline {
	BlockMatchingNode::BlockMatchingNode(NodeId id, const BlockMatchingNodeParams& params)
		: NodeBase(id, "BlockMatchingNode", params.listener)
		, m_params(id, "blockMatchingParams", params.registry, *this)
		, m_prev(id, "prev.greyscale", params.registry, params.resRegistry, params.prevImageDesc)
		, m_curr(id, "curr.greyscale", params.registry, params.resRegistry, params.currImageDesc)
		, m_stats(id, "blockMatching.stats", params.registry, params.resRegistry, params.statsDesc)
	{
	}

	void BlockMatchingNode::Execute(const CudaExecutionContext& ctx) {
		cuda::KernelContext kernelCtx {
			ctx.stream,
			ctx.profiler
		};

		auto outView = m_stats.View();

		cuda::motion::BlockMatching(
			m_prev.View(),
			m_curr.View(),
			outView,
			m_params.Value(),
			kernelCtx
		);
	}
}
