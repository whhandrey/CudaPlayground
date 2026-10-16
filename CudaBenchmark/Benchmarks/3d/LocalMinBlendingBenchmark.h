#pragma once

namespace bench::localmin_blending3d {
	void Blending3dBenchmark();
	void LocalMin3dBenchmark();

	void LocalMin3dThenBlend3dBenchmark();
	void LocalMin3dWithBlend3dFusedBenchmark();

	void LocalMin3dFusedSeparableBenchmark();
	void LocalMin3dFusedHaloStreamZBenchmark(int planesPerBatch);

	void LocalMin3dFusedSepMultipleOutputsBenchmark(int outputsPerThread);
	void LocalMin3dFusedSepMultipleOutputsV2Benchmark(int outputsPerThread);

	void LocalMin3dFusedHaloYMultipleOutputsBenchmark(int outputsPerThread);

	void LocalMin3dOptThenBlend3dBenchmark();
	void LocalMin3dOptWithBlend3dFusedBenchmark();
}
