#pragma once
#include "../Port/IPortListener.h"
#include "../Context/CudaExecutionContext.h"
#include "../Graph/NodeChangeListener.h"
#include "../Common/Id.h"
#include <memory>

namespace dataflow {
	class INode : public IPortListener {
	public:
		virtual ~INode() = default;

		virtual void Execute(const CudaExecutionContext& ctx) = 0;
	};

	class NodeBase : public INode {
	public:
		using Ptr = std::unique_ptr<NodeBase>;

		NodeBase(NodeId id, const std::string& name, INodeChangeListener& listener);

		NodeId Id() const;
		std::string Name() const;

	protected:
		void OnInputChanged(PortId id) override;

	protected:
		const NodeId m_id;
		const std::string m_name;

		INodeChangeListener& m_listener;
	};
}
