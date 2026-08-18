#include "ResourceRegistry.h"
#include <algorithm>
#include <iterator>

namespace processing::resource {
	ResourceId ResourceRegistry::RegisterRequest(const std::string& name, const ResourceDesc& desc) {
		const auto it = std::find_if(m_requests.begin(), m_requests.end(), [&name, &desc](const auto& entry) {
			return name == entry.name && desc == entry.desc;
		});

		if (it != m_requests.end()) {
			return it->id;
		}

		const auto entry = ResourceEntry { m_nextResId, name, desc };
		m_requests.emplace_back(entry);

		return m_nextResId++;
	}

	void ResourceRegistry::ClearRequests() {
		m_requests.clear();
		m_nextResId = 0;
	}

	std::vector<ResourceRequest> ResourceRegistry::BuildRequests() const {
		std::vector<ResourceRequest> output;
		output.reserve(m_requests.size());

		std::transform(m_requests.begin(), m_requests.end(), std::back_inserter(output), [](const auto& req) {
			return ResourceRequest {
				req.id,
				req.desc.domain,
				req.desc.sampleType,
				req.desc.sizeOfElemBytes,
				req.desc.dim
			};
		});

		return output;
	}

	void ResourceRegistry::SupplyResources(std::map<ResourceId, UntypedResourceView>&& resources) {
		m_resourceViews = std::move(resources);
	}

	UntypedResourceView ResourceRegistry::ResolveRaw(ResourceId id) const {
		const auto it = m_resourceViews.find(id);
		if (it == m_resourceViews.end()) {
			throw std::logic_error("ResourceRegistry::ResolveRaw: requested unknown resource");
		}

		return it->second;
	}
}
