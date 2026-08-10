#include "Node.h"

namespace dataflow {
	NodeBase::NodeBase(NodeId id, INodeChangeListener& listener)
		: m_id{ id }
		, m_listener{ listener }
	{
	}

	// Id not yet used for now
	void NodeBase::OnInputChanged(PortId /*id*/) {
		m_listener.OnNodeChanged(m_id);
	}
}
