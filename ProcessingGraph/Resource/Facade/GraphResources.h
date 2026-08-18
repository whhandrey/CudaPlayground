#pragma once
#include "../Id.h"
#include "../Workspace/CudaWorkspace.h"
#include "../Planner/GpuResourceLayoutPlanner.h"
#include "../Registry/ResourceRequest.h"
#include "../../Memory/PinnedImagePool.h"
#include <Image/ImageTypes.h>
#include <vector>
#include <map>
#include <span>

namespace processing::resource {
	using cuda::memory::PinnedImagePool;

	class GraphResources {
	public:
		GraphResources();

	public:
		std::map<ResourceId, UntypedResourceView> Build(const std::vector<ResourceRequest>& requests);

	private:
		std::map<ResourceId, UntypedResourceView> MakePinnedCpuViews(const std::vector<ResourceRequest>& requests);

	private:
		CudaWorkspace m_workspace;
		GpuResourceLayoutPlanner m_gpuResPlanner;

		PinnedImagePool<image::vec4uc> m_pinnedMem;
	};
}
