#pragma once
#include <string>
#include <utility>

namespace motion::debug {
	class IFormatter {
	public:
		virtual ~IFormatter() = default;

		virtual std::string Format(int value) = 0;
		virtual std::string Format(float value) = 0;
		virtual std::string Format(std::pair<int, int> value) = 0;
	};
}
