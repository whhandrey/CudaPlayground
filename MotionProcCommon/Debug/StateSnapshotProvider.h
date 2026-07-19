#pragma once
#include "StateField.h"
#include <vector>
#include <string>

namespace motion::debug {
	class IStateSnapshotProvider {
	public:
		virtual ~IStateSnapshotProvider() = default;

		virtual std::vector<StateField> GetAllFields() = 0;
		virtual std::string GetField(const StateField& field) = 0;
	};
}
