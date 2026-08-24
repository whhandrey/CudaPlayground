#pragma once
#include "NodeChangeListener.h"
#include "../Context/CudaExecutionContext.h"
#include "../Node/Node.h"
#include <memory>
#include <vector>
#include <set>

namespace dataflow {
	class GraphComposer;

	class Graph : public INodeChangeListener {
		friend class GraphComposer;

	private:
		explicit Graph(std::vector<NodeBase::Ptr> nodes, std::vector<INode*> executionOrder);

	public:
		using Ptr = std::unique_ptr<Graph>;

		static Ptr CreateEmpty();

		Graph(Graph&&) noexcept = default;
		Graph& operator=(Graph&&) noexcept = default;

		Graph(const Graph&) = delete;
		Graph& operator=(const Graph&) = delete;

		void Execute(const CudaExecutionContext& ctx);

	private:
		void OnNodeChanged(NodeId id) override;

	private:
		std::vector<NodeBase::Ptr> m_nodes;
		std::vector<INode*> m_executionOrder;

		std::set<NodeId> m_dirtyNodes;
	};
}
