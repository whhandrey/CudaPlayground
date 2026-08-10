#pragma once
#include "../Port/IPortOwner.h"
#include "../Context/ExecutionContext.h"

namespace dataflow {
	using NodeId = size_t;

	class INode : public IPortOwner {
	public:
		virtual ~INode() = default;

		virtual void Execute(ExecutionContext& ctx) = 0;
	};

	class NodeBase : public INode {
	public:
		NodeBase(NodeId id);
	};
}
