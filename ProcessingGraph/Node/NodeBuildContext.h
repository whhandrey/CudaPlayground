#pragma once

namespace processing::resource {
	class IResourceRegistry;
}

namespace dataflow {
	using processing::resource::IResourceRegistry;

	class INodeChangeListener;
	class IPortRegistry;

	struct NodeBuildContext {
		INodeChangeListener& listener;
		IPortRegistry& registry;
		IResourceRegistry& resRegistry;
	};
}
