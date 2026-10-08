#pragma once

namespace bench::localmin_blending3d {
	void Blending3dBenchmark();
	void LocalMin3dBenchmark();
	void LocalMin3dThenBlend3dBenchmark();

	void LocalMin3dFusedSeparableBenchmark();
}
