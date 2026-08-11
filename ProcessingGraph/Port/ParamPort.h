#pragma once
#include "PortBase.h"
#include "IPortListener.h"
#include <vector>
#include <stdexcept>

namespace dataflow {
	template <class T>
	class InParamPort : public PortBase {
	public:
		explicit InParamPort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IPortListener& listener)
			: PortBase(ownerId, name, registry)
			, m_listener{ listener }
		{
		}

		void Update(const T& value) {
			if (m_value != value) {
				m_value = value;
				m_listener.OnInputChanged(Id());
			}
		}

		const T& Value() const {
			return m_value;
		}

		PortDirection Direction() const override {
			return PortDirection::Input;
		}

		PortCategory Category() const override {
			return PortCategory::Param;
		}

	private:
		T m_value;
		IPortListener& m_listener;
	};

	template <class T>
	class OutParamPort : public PortBase {
	public:
		explicit OutParamPort(NodeId ownerId, const std::string& name, IPortRegistry& registry)
			: PortBase(ownerId, name, registry)
		{
		}

		void Publish(const T& value) {
			auto subs = m_registry.GetConnections(Id());

			for (auto* subscriber : subs) {
				auto* inputPort = dynamic_cast<InParamPort<T>*>(subscriber);

				if (!inputPort) {
					throw std::logic_error("OutputParamPort: registry contains an incompatible connection");
				}

				inputPort->Update(value);
			}
		}

		PortDirection Direction() const override {
			return PortDirection::Output;
		}

		PortCategory Category() const override {
			return PortCategory::Param;
		}
	};
}
