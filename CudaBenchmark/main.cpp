#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"
#include "Benchmarks/3d/LocalMinBlendingBenchmark.h"

int main() {
	bench::SobelMag3dFusedSepMultipleOutputsV2Bench(17);
	return 0;
}
