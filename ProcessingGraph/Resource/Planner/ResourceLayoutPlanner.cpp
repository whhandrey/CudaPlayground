#include "ResourceLayoutPlanner.h"
#include <Cuda/KernelCommon.h>

namespace {
	size_t DivUp(size_t a, size_t b) {
		return (a + b - 1) / b;
	}

	size_t AlignUp(size_t value, size_t alignment) {
		return DivUp(value, alignment) * alignment;
	}
}

namespace processing::resource {
	ResourceLayoutPlanner::ResourceLayoutPlanner(size_t resAlignment, size_t pitchAlignment)
		: m_resAlignment{ resAlignment }
		, m_pitchAlignment{ pitchAlignment }
	{
	}

	ResourceAllocation ResourceLayoutPlanner::PlanResource(const ResourceRequest& request, size_t offsetBytes) const {
		const size_t rowBytes = request.dim.x * request.sizeOfElemBytes;
		const size_t pitchBytes = AlignUp(rowBytes, m_pitchAlignment);

		const size_t sizeBytes = pitchBytes * request.dim.y;

		return {
			request.id,
			request.dim,
			request.sampleType,
			request.sizeOfElemBytes,
			offsetBytes,
			pitchBytes,
			sizeBytes
		};
	}

	WorkspaceLayout ResourceLayoutPlanner::Build(std::span<ResourceRequest> requests) const {
		if (requests.empty()) {
			return {};
		}

		std::vector<ResourceAllocation> allocations;
		allocations.reserve(requests.size());

		size_t nextOffsetBytes = 0;

		for (const auto& req : requests) {
			const auto allocation = PlanResource(req, nextOffsetBytes);
			nextOffsetBytes = AlignUp(nextOffsetBytes + allocation.sizeBytes, m_resAlignment);

			allocations.push_back(allocation);
		}

		const size_t requiredBytes = allocations.back().offsetBytes + allocations.back().sizeBytes;

		return {
			std::move(allocations),
			requiredBytes
		};
	}
}
