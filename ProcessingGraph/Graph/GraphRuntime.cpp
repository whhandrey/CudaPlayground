#include "GraphRuntime.h"
#include "GraphComposer.h"

namespace dataflow {
	void GraphRuntime::Construct() {
		auto composer = GraphComposer(Graph::CreateEmpty(), *m_portGraph, *m_resRegistry);

		//composer.AddNode();
		//composer.AddNode();

		auto graph = std::move(composer).Compile();

		// must be done before graph processing
		const auto requests = m_resRegistry->BuildRequests();
		m_resRegistry->SupplyResources(m_resources->Allocate(requests));

		// everything has been built successfully
		// thus move at this point
		m_graph = std::move(graph);
	}
}
