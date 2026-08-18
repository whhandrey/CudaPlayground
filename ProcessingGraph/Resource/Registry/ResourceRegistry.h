#pragma once
#include <vector>
#include <string>
#include <map>
#include "IResourceRegistry.h"
#include "ResourceRequest.h"

namespace processing::resource {
	class ResourceRegistry : public IResourceRegistry {
	public:
		ResourceId RegisterRequest(const std::string& name, const ResourceDesc& desc) override;
		void ClearRequests();

		std::vector<ResourceRequest> BuildRequests() const;
		void SupplyResources(std::map<ResourceId, UntypedResourceView>&& resources);

	private:
		UntypedResourceView ResolveRaw(ResourceId id) const override;

	private:
		struct ResourceEntry {
			ResourceId id;
			std::string name;
			ResourceDesc desc;
		};

		ResourceId m_nextResId{ 0 };
		std::vector<ResourceEntry> m_requests;

		std::map<ResourceId, UntypedResourceView> m_resourceViews;
	};
}
