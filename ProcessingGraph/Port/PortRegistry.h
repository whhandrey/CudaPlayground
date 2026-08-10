#pragma once
#include <vector>
#include <string>
#include "../Common/Id.h"

namespace dataflow {
	class PortBase;

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
}
