#pragma once
#include "../Port/Owner.h"

namespace dataflow {
	class INode : public IPortOwner {
	public:
		virtual ~INode() = default;

		virtual void Execute() = 0;
	};
}
