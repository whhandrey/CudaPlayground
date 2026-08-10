#pragma once
#include "IPortListener.h"
#include "IPortRegistry.h"
#include "../Common/Id.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace dataflow {
	enum class PortType {
		Input,
		Output
	};

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

		virtual PortType Type() const = 0;

	protected:
		const PortId m_portId;
		const NodeId m_ownerId;

		const std::string m_name;
		IPortRegistry& m_registry;
	};

	template <class T>
	class InputPort : public PortBase {
	public:
		explicit InputPort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IPortListener& listener)
			: PortBase(ownerId, name, registry)
			, m_listener{ listener }
		{
		}

		void Update(const T& value) {
			if (m_value != value) {
				m_value = value;
				m_listener.OnInputChanged();
			}
		}

		const T& Value() const {
			return m_value;
		}

		PortType Type() const override {
			return PortType::Input;
		}

	private:
		T m_value;
		IPortListener& m_listener;
	};

	template <class T>
	class OutputPort : public PortBase {
	public:
		explicit OutputPort(NodeId ownerId, const std::string& name, IPortRegistry& registry)
			: PortBase(ownerId, name, registry)
		{
		}

		void Publish(const T& value) {
			auto subs = m_registry.GetConnections(Id());

			for (auto* subscriber : subs) {
				auto* inputPort = dynamic_cast<InputPort<T>*>(subscriber);

				if (!inputPort) {
					throw std::logic_error("OutputPort: registry contains an incompatible connection");
				}

				inputPort->Update(value);
			}
		}

		PortType Type() const override {
			return PortType::Output;
		}
	};
}
