#pragma once
#include "../Id.h"
#include <Image/ImageTypes.h>
#include <vector>
#include <span>

namespace processing::resource {
	struct ResourceAllocation {
		ResourceId id;
		
		image::vec2ui dim;
		std::type_index sampleType;
		size_t sizeOfElemBytes;

		size_t offsetBytes;
		size_t pitchBytes;
		size_t sizeBytes;
	};

	struct WorkspaceLayout {
		std::vector<ResourceAllocation> allocations;
		size_t requiredBytes;
	};
}
