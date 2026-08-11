#pragma once
#include "../../Port/ParamPort.h"
#include "../../Node/Node.h"
#include "../../Port/IPortRegistry.h"
#include <Cuda/Motion/BlockMatching.h>

namespace pipeline {
	using cuda::motion::BlockMatchingParams;
	using dataflow::NodeId;
	using dataflow::CudaExecutionContext;

	class BlockMatchingNode : public dataflow::NodeBase {
	public:
		BlockMatchingNode(NodeId id, dataflow::INodeChangeListener& listener, dataflow::IPortRegistry& registry);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		dataflow::InParamPort<BlockMatchingParams> m_params;
	};
}
