#pragma once
#include "ViewType.h"
#include <memory>

namespace cuda::motion::render {
	class IMotionViewRenderer {
	public:
		using Ptr = std::unique_ptr<IMotionViewRenderer>;

		virtual ~IMotionViewRenderer() = default;
		virtual void Render() = 0;

		static Ptr Create(ViewType type);
	};
}
