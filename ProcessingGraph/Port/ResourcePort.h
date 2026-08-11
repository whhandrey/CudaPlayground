#pragma once
#include "PortBase.h"
#include "IPortListener.h"
#include "../ResourceStore/IResourceStore.h"
#include "../Common/Id.h"
#include <vector>
#include <stdexcept>

namespace dataflow {
	using cuda::memory::IResourceStore;

	template <class T>
	class ResourceInPort : public PortBase {
	public:
		ResourceInPort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IResourceStore& store)
			: PortBase(ownerId, name, registry)
			, m_store{ store }
		{
		}

		void Update(const T& resource) {
			if (m_resource != resource) {
				m_resource = resource;
			}
		}

		const T& Resource() const {
			return m_resource;
		}

		PortDirection Direction() const override {
			return PortDirection::Input;
		}

		PortCategory Category() const override {
			return PortCategory::Resource;
		}

	private:
		T m_resource;
		IResourceStore& m_store;
	};

	template <class T>
	class ResourceOutPort : public PortBase {
	public:
		explicit ResourceOutPort(NodeId ownerId, const std::string& name, IPortRegistry& registry)
			: PortBase(ownerId, name, registry)
		{
		}

		void Publish(const T& resource) {
			auto subs = m_registry.GetConnections(Id());

			for (auto* subscriber : subs) {
				auto* resourceInPort = dynamic_cast<ResourceInPort<T>*>(subscriber);

				if (!resourceInPort) {
					throw std::logic_error("ResourceoutPort: registry contains an incompatible connection");
				}

				resourceInPort->Update(resource);
			}
		}

		PortDirection Direction() const override {
			return PortDirection::Output;
		}

		PortCategory Category() const override {
			return PortCategory::Resource;
		}
	};
}
