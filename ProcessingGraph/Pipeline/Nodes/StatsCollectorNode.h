#pragma once
#include "../../Port/ParamPort.h"
#include "../../Node/Node.h"
#include "../../Port/IPortRegistry.h"
#include <Cuda/Motion/BlockMatching.h>
#include <Motion/Debug/Stats.h>

namespace pipeline {
	using cuda::motion::BlockMatchingParams;
	using cuda::motion::BlockMatchStats;
	using namespace dataflow;

	class StatsCollectorNode : public NodeBase {
	public:
		StatsCollectorNode(NodeId id, INodeChangeListener& listener, IPortRegistry& registry);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		InParamPort<BlockMatchingParams> m_params;
		OutParamPort<motion::debug::StatsPacket> m_statsOut;
	};
}
