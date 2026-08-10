#pragma once
#include "../Port/IPortListener.h"
#include "../Context/CudaExecutionContext.h"
#include "../Graph/NodeChangeListener.h"
#include "../Common/Id.h"

namespace dataflow {
	class INode : public IPortListener {
	public:
		virtual ~INode() = default;

		virtual void Execute(const CudaExecutionContext& ctx) = 0;
	};

	class NodeBase : public INode {
	public:
		NodeBase(NodeId id, INodeChangeListener& listener);

	protected:
		void OnInputChanged() override;

	protected:
		const NodeId m_id;
		INodeChangeListener& m_listener;
	};
}
