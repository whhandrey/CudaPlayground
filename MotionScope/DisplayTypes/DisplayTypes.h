#pragma once
#include <string>
#include <Qimage>
#include <GpuProcessor/ViewRenderer/ViewType.h>

namespace app {
	using cuda::motion::render::ViewType;

	enum class ViewSlot {
		First,
		Second
	};

	enum class PlayMode {
		Normal,
		Loop
	};

	enum class PlayState {
		Play,
		Pause
	};

	struct ViewOption {
		std::string id;
		std::string label;
	};

	using FramePair = std::pair<int, int>;

	struct SlottedImage {
		ViewSlot slot;
		QImage image;
	};

	struct SlottedView {
		ViewSlot slot;
		ViewType view;
	};
}
