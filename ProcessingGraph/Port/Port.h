#pragma once
#include "Owner.h"
#include <vector>

namespace dataflow {
	template <class T>
	class InputPort {
	public:
		explicit InputPort(IPortOwner& owner)
			: m_owner{ owner }
		{
		}

		void Update(T value) {
			if (m_value != value) {
				m_value = std::move(value);
				m_owner.OnInputChanged();
			}
		}

		const T& Value() const {
			return m_value;
		}

	private:
		T m_value;
		IPortOwner& m_owner;
	};

	template <class T>
	class OutputPort {
	public:
		void AddSubscriber(InputPort<T>& port) {
			m_subscribers.push_back(&port);
		}

		void RemoveSubscriber(InputPort<T>& port) {
			std::erase(m_subscribers, &port);
		}

		void Update(const T& value) {
			for (auto* subscriber : m_subscribers) {
				subscriber->Update(value);
			}
		}

	private:
		std::vector<InputPort<T>*> m_subscribers;
	};
}
