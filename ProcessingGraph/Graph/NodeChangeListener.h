#pragma once
#include "../Common/Id.h"

namespace dataflow {
	class INodeChangeListener {
	public:
		virtual ~INodeChangeListener() = default;
		virtual void OnNodeChanged(NodeId id) = 0;
	};
}
