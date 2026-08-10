#include "Node.h"

namespace dataflow {
	NodeBase::NodeBase(NodeId id, INodeChangeListener& listener)
		: m_id{ id }
		, m_listener{ listener }
	{
	}

	void NodeBase::OnInputChanged()
	{
		m_listener.OnNodeChanged(m_id);
	}
}
