#pragma once
#include <Image/ImageTypes.h>
#include <typeindex>
#include "../Id.h"
#include "../MemoryDomain.h"

namespace processing::resource {
	struct ResourceRequest {
		ResourceId id;
		MemoryDomain domain;
		std::type_index sampleType;
		size_t sizeOfElemBytes;
		image::vec2ui dim;
	};
}
