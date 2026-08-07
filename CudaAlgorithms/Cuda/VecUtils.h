#pragma once

namespace util {
	namespace vec {
		template <class Vec2, class Vec1>
		Vec2 ConvertTo(Vec1 vec1) {
			return { vec1.x, vec1.y };
		}

		template <class Vec1, class Vec2>
		bool Eq(Vec1 a, Vec2 b) {
			return a.x == b.x && a.y == b.y;
		}
	}
}
