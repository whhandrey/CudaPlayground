#include "UploadNode.h"

namespace pipeline {
	UploadNode::UploadNode(NodeId id, dataflow::INodeChangeListener& listener)
		: NodeBase(id, listener)
	{
	}

	void UploadNode::Execute(const CudaExecutionContext& ctx) {
		// code here
	}
}
