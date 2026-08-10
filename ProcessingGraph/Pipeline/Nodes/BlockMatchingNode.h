#pragma once
#include "../../Port/Port.h"
#include "../../Node/Node.h"
#include <Cuda/Motion/BlockMatching.h>

namespace pipeline {
	using cuda::motion::BlockMatchingParams;
	using dataflow::CudaExecutionContext;

	class BlockMatchingNode : public dataflow::NodeBase {
	public:
		BlockMatchingNode(dataflow::NodeId id, dataflow::INodeChangeListener& listener);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		//dataflow::InputPort<BlockMatchingParams> m_params;
	};
}
