#pragma once
#include "PortBase.h"
#include "IPortListener.h"
#include "../Resource/Registry/IResourceRegistry.h"
#include "../Common/Id.h"
#include <vector>
#include <stdexcept>

namespace dataflow {
	using processing::resource::IResourceRegistry;
	using processing::resource::ResourceDesc;

	class ResourceBasePort : public PortBase {
	public:
		ResourceBasePort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IResourceRegistry& resRegistry, const ResourceDesc& desc)
			: PortBase(ownerId, name, registry)
			, m_resRegistry{ resRegistry }
			, m_resourceId{ resRegistry.RegisterRequest(name, desc) }
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
		IResourceRegistry& m_resRegistry;
	};

	template <class ResourceView, PortDirection Dir>
	class ResourcePort : public ResourceBasePort {
	public:
		ResourcePort(NodeId ownerId, const std::string& name, IPortRegistry& registry, IResourceRegistry& resRegistry, const ResourceDesc& desc)
			: ResourceBasePort(ownerId, name, registry, resRegistry, desc)
		{
		}

		ResourceView View() const {
			return m_resRegistry.Resolve<ResourceView>(ResId());
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
