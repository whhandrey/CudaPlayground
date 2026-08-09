#pragma once

namespace dataflow {
	class IPortOwner {
	public:
		virtual ~IPortOwner() = default;

		virtual void OnInputChanged() = 0;
	};
}
