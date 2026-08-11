#pragma once
#include <vector>
#include <string>
#include <stdexcept>

namespace cuda::memory {
	using ResourceId = size_t;

	class IResourceStore {
	public:
		virtual ResourceId Get() = 0;
	};
}
