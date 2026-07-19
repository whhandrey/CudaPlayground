#include "DebugFormatter.h"

namespace app {
	std::string DebugFormatter::Format(int value) const
	{
		return std::to_string(value);
	}

	std::string DebugFormatter::Format(float value) const
	{
		return std::to_string(value);
	}

	std::string DebugFormatter::Format(int first, int second) const
	{
		return "(" + std::to_string(first) + "; " + std::to_string(second) + ")";
	}
}
