#include "IPortRegistry.h"
#include "Port.h"

namespace dataflow {
	class PortRegistry {
	public:
		PortId Register(PortBase& port);
		void Unregister(PortId id);

	private:
		struct Entry {
			PortId m_portId;
			PortBase* m_port;
		};

		std::vector<Entry> m_ports;
		PortId m_nextPortId{ 0 };
	};

	PortId PortRegistry::Register(PortBase& port) {
		const Entry ent = { m_nextPortId, &port };
		m_ports.push_back(ent);

		return m_nextPortId++;
	}

	void PortRegistry::Unregister(PortId id) {

	}
}
