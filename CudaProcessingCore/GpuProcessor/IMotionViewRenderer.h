#pragma once
#include <memory>

namespace cuda::motion::render {
	enum class ViewType {
		ConfidenceMap,
		VisualizationMap
	};

	class IMotionViewRenderer {
	public:
		using Ptr = std::unique_ptr<IMotionViewRenderer>;

		virtual ~IMotionViewRenderer() = default;
		virtual void Render() = 0;

		static Ptr Create(ViewType type);
	};
}
