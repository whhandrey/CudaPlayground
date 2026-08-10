#pragma once
#include <span>
#include "../Common/Id.h"

namespace dataflow {
	class PortBase;

	class IPortRegistry {
	public:
		virtual ~IPortRegistry() = default;

		virtual PortId Register(PortBase& port) = 0;
		virtual void Unregister(PortId id) = 0;

		virtual std::span<PortBase* const> GetConnections(PortId id) const = 0;
	};
}
