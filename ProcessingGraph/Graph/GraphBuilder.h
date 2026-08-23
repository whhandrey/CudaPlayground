#pragma once
#include "../Port/IPortRegistry.h"
#include "../Resource/Registry/ResourceRegistry.h"
#include "PortGraph.h"
#include "../Node/Node.h"
#include "../Node/NodeBuildContext.h"
#include <vector>
#include <memory>

namespace dataflow {
	using processing::resource::IResourceRegistry;
	using processing::resource::ResourceRegistry;

	class GraphBuilder {
	public:
		template <class Node, class... Args>
		void AddNode(const std::string& name, Args&&... args) {
			static_assert(std::derived_from<Node, INode>);

			NodeBuildContext context{
				.registry = &m_portGraph,
				.resRegistry = &m_resourceReg
			};

			NodeEntry entry {
				.id = m_nextId,
				.nodeName = name,
				.node = std::make_unique<Node>(m_nextId, std::forward<Args>(args)...)
			};

			++m_nextId;
			m_nodes.emplace_back(std::move(entry));
		}

	private:
		struct NodeEntry {
			NodeId id;
			std::string nodeName;
			std::unique_ptr<INode> node;
		};

		NodeId m_nextId = 0;
		std::vector<NodeEntry> m_nodes;

		PortGraph m_portGraph;
		ResourceRegistry m_resourceReg;
	};
}
