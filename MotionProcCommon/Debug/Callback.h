#pragma once
#include "StatsProvider.h"
#include <functional>

namespace motion::debug {
	using StatsCallback = std::function<void(IStatsProvider::Ptr)>;
}
