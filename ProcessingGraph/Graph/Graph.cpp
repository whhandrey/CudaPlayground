#include "Graph.h"

namespace dataflow {
	Graph::Graph(std::vector<NodeBase::Ptr> nodes, std::vector<INode*> executionOrder)
		: m_nodes{ std::move(nodes) }
		, m_executionOrder{ std::move(executionOrder) }
	{
	}

	Graph::Ptr Graph::CreateEmpty() {
		return std::unique_ptr<Graph>(new Graph({}, {}));
	}

	void Graph::Execute(const CudaExecutionContext& ctx) {
		for (auto* node : m_executionOrder) {
			node->Execute(ctx);
		}
	}

	void Graph::OnNodeChanged(NodeId id) {
		m_dirtyNodes.insert(id);
	}
}
