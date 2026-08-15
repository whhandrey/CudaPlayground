#pragma once
#include "../Id.h"
#include <Image/ImageTypes.h>
#include <vector>
#include <span>

namespace processing::resource {
	struct ResourceRequest {
		ResourceId id;
		size_t elementSizeBytes;
		image::vec2ui dim;
	};

	struct ResourceAllocation {
		ResourceId id;
		size_t offsetBytes;
		size_t pitchBytes;
		size_t sizeBytes;
	};

	struct WorkspaceLayout {
		std::vector<ResourceAllocation> allocations;
		size_t requiredBytes;
	};

	class ResourceLayoutPlanner {
	public:
		ResourceLayoutPlanner(size_t resAlignment, size_t pitchAlignment);

	public:
		WorkspaceLayout Build(std::span<ResourceRequest> requests) const;

	private:
		ResourceAllocation PlanResource(const ResourceRequest& request, size_t offsetBytes) const;

	private:
		const size_t m_resAlignment;
		const size_t m_pitchAlignment;
	};
}
