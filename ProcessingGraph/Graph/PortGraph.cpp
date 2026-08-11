#include "PortGraph.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace {
	struct PortConnectionKey {
		dataflow::PortCategory category;
		std::string name;

		auto operator<=>(const PortConnectionKey&) const = default;
	};

	struct PortGroup {
		std::vector<dataflow::PortBase*> inputs;

		// There is always one unique OutputPort with Key(name, category)
		// std::vector here is to catch duplicates at runtime and throw if they are found
		std::vector<dataflow::PortBase*> outputs;
	};
}

namespace dataflow {
	PortId PortGraph::Register(PortBase& port) {
		m_ports.emplace_back(&port);
		return ++m_nextPortId;
	}

	void PortGraph::Unregister(PortId id) {
		auto portIt = std::find_if(m_ports.begin(), m_ports.end(), [id](auto* port) {
			return port->Id() == id;
		});

		if (portIt == m_ports.end()) {
			return;
		}

		auto* port = *portIt;

		// If port is outputPort, remove its inputPorts subscribers
		m_connections.erase(id);

		// If port is an input, remove it from every output
		// It scans all outputIds but since vectors are small it is not a problem
		for (auto& [outputId, inputs] : m_connections) {
			std::erase(inputs, port);
		}

		// Erase from generic port storage
		m_ports.erase(portIt);
	}

	std::span<PortBase* const> PortGraph::GetConnections(PortId id) const {
		const auto it = m_connections.find(id);
		if (it == m_connections.end()) {
			throw std::logic_error("PortGraph::GetConnections: no output port connected with PortId: " + std::to_string(id));
		}

		return it->second;
	}

	void PortGraph::ConnectPorts() {
		m_connections.clear();

		std::map<PortConnectionKey, PortGroup> groups;

		// Group every port in one pass.
		for (auto* port : m_ports) {
			PortConnectionKey key {
				port->Category(),
				port->Name()
			};

			auto& group = groups[key];

			switch (port->Direction()) {
			case PortDirection::Input:
				group.inputs.push_back(port);
				break;

			case PortDirection::Output:
				group.outputs.push_back(port);
				break;
			}
		}

		// Create output -> inputs connections.
		for (auto& [key, group] : groups) {

			if (group.outputs.empty()) {
				throw std::logic_error("PortGraph: inputs have no corresponding output");
			}

			if (group.outputs.size() > 1) {
				throw std::logic_error("Multiple output ports have the same category and name");
			}

			auto* outputPort = group.outputs.front();
			m_connections.emplace(outputPort->Id(), std::move(group.inputs));
		}
	}
}
