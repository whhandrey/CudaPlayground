#pragma once
#include "Stats.h"
#include <functional>

namespace motion::debug {
	using StatsCallback = std::function<void(StatsPacket&&)>;
}
