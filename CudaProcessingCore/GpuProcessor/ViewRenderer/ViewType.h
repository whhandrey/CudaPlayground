#pragma once
#include <vector>

namespace cuda::motion::render {
	enum class ViewType {
		ConfMap,
		MotionMap,
		MagnitudeMap,
		ArrowsMap
	};

	inline std::vector<ViewType> AllViews() {
		return {
			ViewType::ConfMap,
			ViewType::MotionMap,
			ViewType::MagnitudeMap,
			ViewType::ArrowsMap
		};
	}
}
