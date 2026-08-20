#include "Node.h"

namespace dataflow {
	NodeBase::NodeBase(NodeId id, const std::string& name, INodeChangeListener& listener)
		: m_id{ id }
		, m_name{ name }
		, m_listener{ listener }
	{
	}

	NodeId NodeBase::Id() const {
		return m_id;
	}

	std::string NodeBase::Name() const {
		return m_name;
	}

	// Id not yet used for now
	void NodeBase::OnInputChanged(PortId /*id*/) {
		m_listener.OnNodeChanged(Id());
	}
}
