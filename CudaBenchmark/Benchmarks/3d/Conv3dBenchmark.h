#pragma once

namespace bench {
	enum class Conv3dType {
		FusedSeparable,
		FusedSeparableMultipleOutputs
	};

	void Conv3dCheckIfMatchWithNaive(Conv3dType type);

	void Conv3dNaiveSharedMemBench();
	void Conv3dFusedSeparableBench();
	void Conv3dFusedSeparableMultipleOutputsBench(int numOutputs);
}
