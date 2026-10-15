#pragma once

namespace bench {
	void SobelMag3dNaiveBench();

	void SobelMag3dNaiveSharedMemBench();
	void SobelMag3dFusedSeparableBench();

	void SobelMag3dFusedSepMultipleOutputsBench(int outputsPerThread);
	void SobelMag3dFusedSepMultipleOutputsV2Bench(int outputsPerThread);

	void CheckIfSepMultMatchesNaive();
}
