#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include "IResourceRegistry.h"

namespace processing::resource {
	class ResourceRegistry : public IResourceRegistry {
	public:
		ResourceId RegisterRequest(const std::string& name, const ResourceDesc& desc) override;

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
