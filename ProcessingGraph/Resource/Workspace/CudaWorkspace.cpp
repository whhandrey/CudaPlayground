#include "CudaWorkspace.h"
#include <algorithm>

namespace processing::workspace {
	std::vector<ResourceView> CudaWorkspace::GetGpuResourceViews(const resource::WorkspaceLayout& layout) {
		if (m_mem.SizeBytes() < layout.requiredBytes) {
			m_mem.Allocate(layout.requiredBytes);
		}

		std::vector<ResourceView> output;
		output.reserve(layout.allocations.size());

		std::transform(layout.allocations.begin(), layout.allocations.end(), output.begin(), [this](const auto& alloc) {
			resource::UntypedResourceView view {
				static_cast<unsigned char*>(m_mem.Mem()) + alloc.offsetBytes,
				alloc.dim,
				alloc.pitchBytes,
				resource::MemoryDomain::CudaDevice,
				alloc.sampleType
			};

			return ResourceView{ alloc.id, view };
		});

		return output;
	}
}
