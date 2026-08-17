#pragma once
#include "../Id.h"
#include "../Registry/ResourceRequest.h"
#include "WorkspaceLayout.h"
#include <Image/ImageTypes.h>
#include <vector>
#include <span>

namespace processing::resource {
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
