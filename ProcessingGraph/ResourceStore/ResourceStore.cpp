#include "ResourceStore.h"

namespace cuda::memory {
	ResourceId ResourceStore::RegisterRequest(const std::string& name, const ResourceDesc& desc) {
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

	UntypedResourceView ResourceStore::ResolveRaw(ResourceId id) const {
		return {};
		//auto& entry = m_resources.at(id);

		//return {
		//	.data = entry.Data(),
		//	.dim = entry.Dimensions(),
		//	.strideBytes = entry.StrideBytes(),
		//	.domain = entry.Domain(),
		//	.sampleType = entry.SampleType()
		//};
	}
}
