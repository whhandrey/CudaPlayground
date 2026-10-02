#pragma once

namespace bench {
	void Conv3dNaiveSharedMemBench();
	void Conv3dFusedSeparableBench();
	void Conv3dFusedSeparableMultipleOutputsBench(int numOutputs);
}
