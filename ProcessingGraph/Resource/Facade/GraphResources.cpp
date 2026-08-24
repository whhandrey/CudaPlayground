#include "GraphResources.h"
#include <algorithm>

namespace {
	constexpr image::vec2ui MaxDimPinned{ 3840, 2160 };
	constexpr size_t SlotCountPinned = 5;

	constexpr size_t GpuResAlignment = 256;
	constexpr size_t GpuPitchAlignment = 32;
}

namespace {
	using processing::resource::MemoryDomain;
	using processing::resource::UntypedResourceView;
	using processing::resource::ResourceRequest;

	UntypedResourceView ToRawView(const image::PinnedImageView<image::vec4uc>& view) {
		return {
			static_cast<void*>(view.m_ptr),
			view.m_dim,
			view.m_pitch,
			MemoryDomain::HostPinned,
			typeid(image::vec4uc)
		};
	}

	std::map<MemoryDomain, std::vector<ResourceRequest>> SplitRequestsByDomain(const std::vector<ResourceRequest>& requests) {
		std::map<MemoryDomain, std::vector<ResourceRequest>> output;

		for (const auto& req : requests) {
			output[req.domain].push_back(req);
		}

		return output;
	}
}

namespace processing::resource {
	GraphResources::GraphResources()
		: m_gpuResPlanner(GpuResAlignment, GpuPitchAlignment)
		, m_pinnedMem(MaxDimPinned, SlotCountPinned)
	{
	}

	std::map<ResourceId, UntypedResourceView> GraphResources::Allocate(const std::vector<ResourceRequest>& requests) {
		std::map<ResourceId, UntypedResourceView> output;

		const auto reqMap = SplitRequestsByDomain(requests);

		if (reqMap.contains(MemoryDomain::Host)) {
			throw std::logic_error("GraphResources::Build: Host resources are not supported");
		}

		if (reqMap.contains(MemoryDomain::HostPinned)) {
			output.merge(MakePinnedCpuViews(reqMap.at(MemoryDomain::HostPinned)));
		}

		if (reqMap.contains(MemoryDomain::CudaDevice)) {
			const auto wsLayout = m_gpuResPlanner.Build(reqMap.at(MemoryDomain::CudaDevice));
			output.merge(m_workspace.BuildGpuResources(wsLayout));
		}

		return output;
	}

	std::map<ResourceId, UntypedResourceView> GraphResources::MakePinnedCpuViews(const std::vector<ResourceRequest>& requests) {
		if (requests.size() > m_pinnedMem.SlotCount()) {
			throw std::logic_error("GraphResources::BuildPinnedCpuViews: requested more CPU resources than allocated");
		}

		std::map<ResourceId, UntypedResourceView> output;

		for (size_t i = 0; i < requests.size(); ++i) {
			if (requests[i].sampleType != typeid(image::vec4uc)) {
				throw std::logic_error("GraphResources::BuildPinnedCpuViews: requested resource with unsupported type");
			}

			auto view = ToRawView(m_pinnedMem.View(requests[i].dim, i));
			output.emplace(requests[i].id, view);
		}

		return output;
	}
}
