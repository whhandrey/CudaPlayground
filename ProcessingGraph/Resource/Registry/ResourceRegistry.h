#pragma once
#include <vector>
#include <string>
#include "IResourceRegistry.h"
#include "ResourceRequest.h"

namespace processing::resource {
	class ResourceRegistry : public IResourceRegistry {
	public:
		ResourceId RegisterRequest(const std::string& name, const ResourceDesc& desc) override;
		std::vector<ResourceRequest> BuildRequests(MemoryDomain domain) const;

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
	};
}
