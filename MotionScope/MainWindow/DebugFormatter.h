#pragma once
#include <Debug/Formatter.h>

namespace app {
	using ::motion::debug::IFormatter;

	class DebugFormatter : public IFormatter {
	public:
		std::string Format(int value) const override;
		std::string Format(float value) const override;
		std::string Format(int first, int second) const override;
	};
}
