#pragma once
#include "../Port/IPortRegistry.h"
#include "../Port/Port.h"
#include <vector>
#include <map>

namespace dataflow {
	class PortGraph : public IPortRegistry {
	public:
		PortId Register(PortBase& port) override;
		void Unregister(PortId id) override;

		std::span<PortBase* const> GetConnections(PortId id) const override;

		void ConnectPorts();

	private:
		std::vector<PortBase*> m_ports;

		std::map<PortId, std::vector<PortBase*>> m_connections;
		PortId m_nextPortId{ 0 };
	};
}
