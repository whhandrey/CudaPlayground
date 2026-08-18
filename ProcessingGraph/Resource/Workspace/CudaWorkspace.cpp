#include "CudaWorkspace.h"
#include <algorithm>
#include <iterator>

namespace processing::resource {
	std::map<ResourceId, UntypedResourceView> CudaWorkspace::BuildGpuResources(const WorkspaceLayout& layout) {
		if (m_mem.SizeBytes() < layout.requiredBytes) {
			m_mem.Allocate(layout.requiredBytes);
		}

		std::map<ResourceId, UntypedResourceView> output;

		std::transform(layout.allocations.begin(), layout.allocations.end(), std::inserter(output, output.end()), [this](const auto& alloc) {
			resource::UntypedResourceView view {
				static_cast<unsigned char*>(m_mem.Mem()) + alloc.offsetBytes,
				alloc.dim,
				alloc.pitchBytes,
				resource::MemoryDomain::CudaDevice,
				alloc.sampleType
			};

			return std::pair{ alloc.id, view };
		});

		return output;
	}
}
