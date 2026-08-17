#pragma once
#include <Image/ImageTypes.h>
#include <typeindex>
#include "../Id.h"

namespace processing::resource {
	struct ResourceRequest {
		ResourceId id;
		std::type_index sampleType;
		size_t sizeOfElemBytes;
		image::vec2ui dim;
	};
}
