#include "Port.h"

namespace dataflow {
	PortBase::PortBase(NodeId ownerId, const std::string& name, PortRegistry& registry)
		: m_portId{ registry.Register(this) }
		, m_ownerId{ ownerId }
		, m_name{ name }
		, m_registry{ registry }
	{
	}

	PortBase::~PortBase() {
		m_registry.Unregister(Id());
	}

	std::string PortBase::Name() const {
		return m_name;
	}

	NodeId PortBase::OwnerId() const {
		return m_ownerId;
	}

	PortId PortBase::Id() const {
		return m_portId;
	}
}
