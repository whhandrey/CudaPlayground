#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include "ResourceDesc.h"
#include "ResourceTraits.h"

namespace cuda::memory {
	class IResourceStore {
	public:
		virtual ~IResourceStore() = default;
		virtual ResourceId RegisterRequest(const std::string& name, const ResourceDesc& desc) = 0;

		template <class ResourceView>
		ResourceView Resolve(ResourceId id) {
			const auto raw = ResolveRaw(id);
			return ResourceViewTraits<ResourceView>::Make(raw);
		}

	protected:
		virtual UntypedResourceView ResolveRaw(ResourceId id) const = 0;
	};
}
