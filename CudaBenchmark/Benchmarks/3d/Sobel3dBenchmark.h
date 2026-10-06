#pragma once

namespace bench {
	void SobelMag3dNaiveSharedMemBench();
	void SobelMag3dFusedSeparableBench();
	void SobelMag3dFusedSepMultipleOutputsBench(int outputsPerThread);

	void CheckIfSepMultMatchesNaive();
}
