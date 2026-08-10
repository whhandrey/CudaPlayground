#pragma once

namespace dataflow {
	class IPortListener {
	public:
		virtual ~IPortListener() = default;
		virtual void OnInputChanged() = 0;
	};
}
