#pragma once
#include <string>
#include <utility>

namespace motion::debug {
	class IFormatter {
	public:
		virtual ~IFormatter() = default;

		virtual std::string Format(int value) const = 0;
		virtual std::string Format(float value) const = 0;
		virtual std::string Format(int first, int second) const = 0;
	};
}
