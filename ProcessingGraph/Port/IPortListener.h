#pragma once
#include "../Common/Id.h"

namespace dataflow {
	class IPortListener {
	public:
		virtual ~IPortListener() = default;
		virtual void OnInputChanged(PortId id) = 0;
	};
}
