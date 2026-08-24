#include "GraphComposer.h"
#include <queue>
#include <cassert>
#include <algorithm>
#include <iterator>

namespace {
    using dataflow::INode;
    using dataflow::NodeBase;
    using dataflow::NodeId;

    std::vector<INode*> ExtractNodesById(const std::vector<NodeId> ids, const std::vector<NodeBase::Ptr>& nodes) {
        std::map<NodeId, NodeBase*> nodesById;

        std::transform(nodes.begin(), nodes.end(), std::inserter(nodesById, nodesById.end()), [](const auto& node) {
            return std::pair{ node->Id(), node.get() };
        });

        std::vector<INode*> output;
        output.reserve(ids.size());

        for (const NodeId id : ids) {
            output.push_back(nodesById.at(id));
        }

        return output;
    }
}

namespace dataflow {
	GraphComposer::GraphComposer(std::unique_ptr<Graph> graph, PortGraph& portGraph, IResourceRegistry& resRegistry)
		: m_graph{ std::move(graph) }
        , m_portGraph{ portGraph }
        , m_resRegistry{ resRegistry }
	{
        if (!m_graph) {
            throw std::logic_error("GraphComposer::Begin: graph is nullptr");
        }

        m_nodes = std::move(m_graph->m_nodes);

        for (const auto& node : m_nodes) {
            m_nextId = std::max(m_nextId, node->Id() + 1);
        }
	}

	void GraphComposer::AddNode(const NodeDefinition& definition) {
		m_nodes.emplace_back(CreateNode(definition));
	}

	void GraphComposer::ReplaceNode(NodeId id, const NodeDefinition& newNodeDef) {
        // todo: impl
	}

    std::unique_ptr<Graph> GraphComposer::Compile() && {
        // must connect ports after registering nodes
        m_portGraph.ConnectPorts();

        const auto dependencies = CreateDependencies();
        auto executionOrder = ExtractNodesById(std::move(TopologicalSort(dependencies)), m_nodes);

        // consructor is friend only for composer, not for unique_ptr
        m_graph = std::unique_ptr<Graph>{ new Graph(std::move(m_nodes), std::move(executionOrder)) };
        return std::move(m_graph);
    }

	NodeBase::Ptr GraphComposer::CreateNode(const NodeDefinition& definition) {
		if (!m_graph) {
			throw std::logic_error("GraphComposer::CreateNode: Begin has not been called");
		}

		const NodeBuildContext context {
			.listener = *m_graph,
			.registry = m_portGraph,
			.resRegistry = m_resRegistry
		};

		return definition.func(m_nextId++, context);
	}

    NodeAdjacency GraphComposer::CreateDependencies() const {
		const auto connections = m_portGraph.GetConnections(PortCategory::Resource);

        NodeAdjacency dependencies;
		for (const auto& [outputPort, inputPorts] : connections) {
			for (const auto& inputPort : inputPorts) {
				dependencies[outputPort->OwnerId()].push_back(inputPort->OwnerId());
			}
		}

		return dependencies;
	}

    std::vector<NodeId> GraphComposer::TopologicalSort(const NodeAdjacency& dependencies) const
    {
        std::map<NodeId, size_t> incomingEdgeCount;

        // Register every node, including completely disconnected nodes.
        for (const auto& node : m_nodes) {
            const auto [_, inserted] = incomingEdgeCount.emplace(node->Id(), 0);

            if (!inserted) {
                throw std::logic_error("GraphComposer::TopologicalSort: duplicate NodeId");
            }
        }

        // Build adjacency and count incoming dependencies.
        for (const auto& [_, consumers] : dependencies) {
            for (const auto& consumerId : consumers) {
                ++incomingEdgeCount.at(consumerId);
            }
        }

        // Nodes without incoming dependencies can execute immediately.
        std::queue<NodeId> readyNodes;

        // Iterate m_nodes to preserve their construction order where possible.
        for (const auto& node : m_nodes) {
            if (incomingEdgeCount.at(node->Id()) == 0) {
                readyNodes.push(node->Id());
            }
        }

        std::vector<NodeId> executionOrder;
        executionOrder.reserve(m_nodes.size());

        while (!readyNodes.empty()) {
            const NodeId nodeId = readyNodes.front();
            readyNodes.pop();

            executionOrder.push_back(nodeId);

            for (const NodeId childId : dependencies.at(nodeId)) {
                auto& childIncomingEdges = incomingEdgeCount.at(childId);

                assert(childIncomingEdges > 0);
                --childIncomingEdges;

                if (childIncomingEdges == 0) {
                    readyNodes.push(childId);
                }
            }
        }

        if (executionOrder.size() != m_nodes.size()) {
            throw std::logic_error("GraphComposer::TopologicalSort: graph contains a cycle");
        }

        return executionOrder;
    }
}
