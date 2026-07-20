#pragma once
#include "StatsField.h"
#include "Formatter.h"
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace motion::debug {
	using FieldGroupMap = std::map<std::string, std::vector<std::string>>;

	class IStatsProvider {
	public:
		using Ptr = std::unique_ptr<IStatsProvider>;

		virtual ~IStatsProvider() = default;

		virtual FieldGroupMap GetFieldNamesByGroups() const = 0;
		virtual std::string GetField(const StatsField& field, IFormatter& formatter) const = 0;
	};
}
