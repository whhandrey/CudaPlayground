#pragma once
#include "../Port/IPortRegistry.h"
#include "../Resource/Registry/ResourceRegistry.h"
#include "PortGraph.h"
#include "../Node/Node.h"
#include "../Node/NodeBuildContext.h"
#include "../Node/NodeDefinition.h"
#include "Graph.h"
#include <vector>
#include <set>

namespace dataflow {
	using processing::resource::IResourceRegistry;
	using NodeAdjacency = std::map<NodeId, std::vector<NodeId>>;

	class GraphComposer {
	public:
		GraphComposer(std::unique_ptr<Graph> graph, PortGraph& portGraph, IResourceRegistry& resRegistry);

		void AddNode(const NodeDefinition& definition);
		void ReplaceNode(NodeId id, const NodeDefinition& newNodeDef);

		std::unique_ptr<Graph> Compile() &&;

	private:
		NodeBase::Ptr CreateNode(const NodeDefinition& definition);
		NodeAdjacency CreateDependencies() const;

		std::vector<NodeId> TopologicalSort(const NodeAdjacency& dependencies) const;

	private:
		std::unique_ptr<Graph> m_graph;

		PortGraph& m_portGraph;
		IResourceRegistry& m_resRegistry;

		NodeId m_nextId = 0;
		std::vector<NodeBase::Ptr> m_nodes;
	};
}
