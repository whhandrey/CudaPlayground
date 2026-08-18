#pragma once
#include "MemoryDomain.h"
#include <Image/ImageTypes.h>
#include <typeindex>
#include <string>

namespace processing::resource {
	struct ResourceDesc {
		image::vec2ui dim;
		size_t sizeOfElemBytes;
		std::type_index sampleType;
		MemoryDomain domain;

		bool operator== (const ResourceDesc& desc) const {
			return dim.x == desc.dim.x
				&& dim.y == desc.dim.y
				&& sizeOfElemBytes == desc.sizeOfElemBytes
				&& sampleType == desc.sampleType
				&& domain == desc.domain;
		}
	};

	template <class SampleType>
	ResourceDesc MakeResourceDesc(const std::string& name, image::vec2ui dim, MemoryDomain domain) {
		return {
			name,
			dim,
			sizeof(SampleType),
			typeid(SampleType),
			domain
		};
	}
}
