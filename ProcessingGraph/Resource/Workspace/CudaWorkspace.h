#pragma once
#include "../../Memory/CudaDeviceMemory.h"
#include "../../Resource/Id.h"
#include "../Registry/UntypedResourceView.h"
#include "../Planner/WorkspaceLayout.h"
#include <map>

namespace processing::resource {
	using cuda::memory::LinearDeviceMemory;

	class CudaWorkspace {
	public:
		std::map<ResourceId, UntypedResourceView> BuildGpuResources(const WorkspaceLayout& layout);

	private:
		LinearDeviceMemory m_mem;
	};
}
