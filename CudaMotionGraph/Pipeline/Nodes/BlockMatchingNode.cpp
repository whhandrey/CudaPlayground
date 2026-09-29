#include "BlockMatchingNode.h"

namespace pipeline {
	BlockMatchingNode::BlockMatchingNode(NodeId id, const NodeBuildContext& ctx, const BlockMatchingNodeParams& params)
		: NodeBase(id, "BlockMatchingNode", ctx.listener)
		, m_params(id, "blockMatchingParams", ctx.registry, *this)
		, m_prev(id, ctx.registry, ctx.resRegistry, params.prevImageDesc)
		, m_curr(id, ctx.registry, ctx.resRegistry, params.currImageDesc)
		, m_stats(id, ctx.registry, ctx.resRegistry, params.statsDesc)
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
