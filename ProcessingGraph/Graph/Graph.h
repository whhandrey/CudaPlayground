#pragma once
#include "NodeChangeListener.h"
#include "../Node/Node.h"
#include <memory>
#include <vector>

namespace dataflow {
	class GraphBuilder;

	class Graph : public INodeChangeListener {
		friend class GraphBuilder;

	private:
		explicit Graph(std::vector<NodeBase::Ptr> nodes);

	public:
		Graph(Graph&&) noexcept = default;
		Graph& operator=(Graph&&) noexcept = default;

		Graph(const Graph&) = delete;
		Graph& operator=(const Graph&) = delete;

		void Execute();

	private:
		void OnNodeChanged(NodeId id) override;

	private:
		std::vector<NodeBase::Ptr> m_nodes;
	};
}
