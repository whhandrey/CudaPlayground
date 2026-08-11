#pragma once
#include "../../Node/Node.h"
#include "../../Port/IPortRegistry.h"
#include <Cuda/Motion/BlockMatching.h>

namespace pipeline {
	using cuda::motion::BlockMatchingParams;
	using dataflow::NodeId;
	using dataflow::CudaExecutionContext;

	class UploadNode : public dataflow::NodeBase {
	public:
		UploadNode(NodeId id, dataflow::INodeChangeListener& listener);

	public:
		void Execute(const CudaExecutionContext& ctx) override;
	};
}
