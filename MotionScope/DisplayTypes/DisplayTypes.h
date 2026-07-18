#pragma once
#include <string>

namespace app {
	enum class ViewSlot {
		View1,
		View2
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
}
