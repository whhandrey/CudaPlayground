#include "StatsCollectorNode.h"

namespace {
	using namespace motion::debug;
	using cuda::motion::BlockMatchingParams;

	void FillParams(const BlockMatchingParams& p, StatsPacket& stats) {
		stats["Algo/Params"].push_back(StatsField{ "BlockDim", p.blockDim, Unit::px });
		stats["Algo/Params"].push_back(StatsField{ "MacroBlockDim", p.macroBlockDim, Unit::px });
		stats["Algo/Params"].push_back(StatsField{ "SearchRad", p.search_halfsize, Unit::px });
	}
}

namespace pipeline {
	StatsCollectorNode::StatsCollectorNode(NodeId id, INodeChangeListener& listener, IPortRegistry& registry)
		: NodeBase(id, "StatsCollectorNode", listener)
		, m_params(id, "blockMatchingParams", registry, *this)
		, m_statsOut(id, "executionStats", registry)
	{
	}

	void StatsCollectorNode::Execute(const CudaExecutionContext& ctx) {
		motion::debug::StatsPacket output;
		FillParams(m_params.Value(), output);

		// TODO: need to improve profiler cause now it only gives time without the node name
		// Actually returns nothing

		m_statsOut.Publish(output);
	}
}
