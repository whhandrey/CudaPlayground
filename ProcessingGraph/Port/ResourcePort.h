#pragma once
#include "PortBase.h"
#include "IPortListener.h"
#include "../ResourceStore/IResourceStore.h"
#include "../Common/Id.h"
#include <vector>
#include <stdexcept>

namespace dataflow {
	using cuda::memory::IResourceStore;
	using cuda::memory::ResourceDesc;

	class ResourceBasePort : public PortBase {
	public:
		ResourceBasePort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IResourceStore& store, const ResourceDesc& desc)
			: PortBase(ownerId, name, registry)
			, m_store{ store }
			, m_resourceId{ store.RegisterRequest(name, desc) }
		{
		}

		PortCategory Category() const override {
			return PortCategory::Resource;
		}

		ResourceId ResId() const {
			return m_resourceId;
		}

	protected:
		const ResourceId m_resourceId;
		IResourceStore& m_store;
	};

	template <class ResourceView, PortDirection Dir>
	class ResourcePort : public ResourceBasePort {
	public:
		ResourcePort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IResourceStore& store, const ResourceDesc& desc)
			: ResourceBasePort(ownerId, name, registry, store, desc)
		{
		}

		ResourceView View() const {
			return m_store.Resolve<ResourceView>(ResId());
		}

		PortDirection Direction() const override {
			return Dir;
		}
	};

	template <class ResourceView>
	using ResourceInPort = ResourcePort<ResourceView, PortDirection::Input>;

	template <class ResourceView>
	using ResourceOutPort = ResourcePort<ResourceView, PortDirection::Output>;
}
