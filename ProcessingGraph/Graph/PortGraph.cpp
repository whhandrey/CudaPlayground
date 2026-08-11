#include "PortGraph.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace {
	std::vector<dataflow::PortCategory> AllCategories() {
		return {
			dataflow::PortCategory::Param,
			dataflow::PortCategory::Resource
		};
	}
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
		const auto categories = AllCategories();

		for (const auto cat : categories) {
			std::vector<PortBase*> outputPorts;

			std::copy_if(m_ports.begin(), m_ports.end(), std::back_inserter(outputPorts), [cat](auto* node) {
				return node->Category() == cat
					&& node->Direction() == PortDirection::Output;
			});

			for (auto* outputPort : outputPorts) {
				const auto portName = outputPort->Name();
				std::vector<PortBase*> inputPorts;

				std::copy_if(m_ports.begin(), m_ports.end(), std::back_inserter(inputPorts), [&portName, cat](auto* node) {
					return node->Category() == cat
						&& node->Direction() == PortDirection::Input
						&& node->Name() == portName;
				});

				const PortId outputId = outputPort->Id();
				if (m_connections.find(outputId) != m_connections.end()) {
					throw std::logic_error("PortGraph::ConnectPorts: output port is already connected");
				}

				m_connections.emplace(outputId, inputPorts);
			}
		}
	}
}
