#include "Graph.h"

namespace dataflow {
	Graph::Graph(std::vector<NodeBase::Ptr> nodes)
		: m_nodes{ std::move(nodes) }
	{
	}

	void Graph::Execute() {

	}

	void Graph::OnNodeChanged(NodeId id) {

	}
}
