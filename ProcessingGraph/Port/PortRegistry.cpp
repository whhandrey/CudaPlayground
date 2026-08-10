#include "PortRegistry.h"
#include "Port.h"

namespace dataflow {
	PortId PortRegistry::Register(PortBase& port) {
		const Entry ent = { m_nextPortId, &port };
		m_ports.push_back(ent);

		return m_nextPortId++;
	}

	void PortRegistry::Unregister(PortId id) {

	}
}
