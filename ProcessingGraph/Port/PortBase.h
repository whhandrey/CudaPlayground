#pragma once
#include "PortType.h"
#include "IPortRegistry.h"
#include "../Common/Id.h"
#include <string>

namespace dataflow {
	class PortBase {
	public:
		PortBase(NodeId ownerId, const std::string& name, IPortRegistry& registry);
		virtual ~PortBase();

		PortBase(const PortBase&) = delete;
		PortBase& operator=(const PortBase&) = delete;
		PortBase(PortBase&&) = delete;
		PortBase& operator=(PortBase&&) = delete;

		std::string Name() const;

		NodeId OwnerId() const;
		PortId Id() const;

		virtual PortDirection Direction() const = 0;
		virtual PortCategory Category() const = 0;

	protected:
		const PortId m_portId;
		const NodeId m_ownerId;

		const std::string m_name;
		IPortRegistry& m_registry;
	};
}
