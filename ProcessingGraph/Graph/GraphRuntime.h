#pragma once
#include "../Resource/Registry/ResourceRegistry.h"
#include "../Node/Node.h"
#include "../Resource/Facade/GraphResources.h"
#include "PortGraph.h"
#include "Graph.h"

namespace dataflow {
	using processing::resource::ResourceRegistry;
	using processing::resource::GraphResources;

	class GraphRuntime {
	public:
		void Construct();

	private:
		std::unique_ptr<PortGraph> m_portGraph;
		std::unique_ptr<ResourceRegistry> m_resRegistry;

		std::unique_ptr<GraphResources> m_resources;

		std::unique_ptr<Graph> m_graph;
	};
}
