#pragma once
#include "../../Memory/CudaDeviceMemory.h"
#include "../../Resource/Id.h"
#include "../Registry/UntypedResourceView.h"
#include "../Planner/WorkspaceLayout.h"
#include <vector>

namespace processing::workspace {
	using cuda::memory::LinearDeviceMemory;
	using resource::ResourceId;

	struct ResourceView {
		ResourceId id;
		resource::UntypedResourceView view;
	};

	class CudaWorkspace {
	public:
		std::vector<ResourceView> GetGpuResourceViews(const resource::WorkspaceLayout& layout);

	private:
		LinearDeviceMemory m_mem;
	};
}
