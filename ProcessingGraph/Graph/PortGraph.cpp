#include "PortGraph.h"
#include <algorithm>
#include <iterator>

namespace dataflow {
	PortId PortGraph::Register(PortBase& port) {
		m_ports.emplace_back(&port);
		return ++m_nextPortId;
	}

	void PortGraph::Unregister(PortId id) {
		// first check if ports is in the generic store
		size_t erasedElems = std::erase_if(m_ports, [id](auto* port) {
			return port->Id() == id;
		});

		if (erasedElems > 0) {
			// then check if the port was outputPort
			m_connections.erase(id);
		}
	}

	std::span<PortBase* const> PortGraph::GetConnections(PortId id) const {
		const auto it = m_connections.find(id);
		if (it == m_connections.end()) {
			throw std::logic_error("PortGraph::GetConnections: no output port connected with PortId: " + std::to_string(id));
		}

		return it->second;
	}

	void PortGraph::ConnectPorts() {
		std::vector<PortBase*> outputPorts;

		std::copy_if(m_ports.begin(), m_ports.end(), std::back_inserter(outputPorts), [](auto* node) {
			return node->Type() == PortType::Output;
		});

		for (auto* outputPort : outputPorts) {
			const auto portName = outputPort->Name();
			std::vector<PortBase*> inputPorts;

			std::copy_if(m_ports.begin(), m_ports.end(), std::back_inserter(inputPorts), [&portName](auto* node) {
				return node->Type() == PortType::Input && node->Name() == portName;
			});

			const PortId outputId = outputPort->Id();
			if (m_connections.find(outputId) != m_connections.end()) {
				throw std::logic_error("PortGraph::ConnectPorts: output port is already connected");
			}

			m_connections.emplace(outputId, inputPorts);
		}
	}
}
