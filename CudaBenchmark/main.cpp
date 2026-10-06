#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"

int main() {
	bench::SobelMag3dFusedSepMultipleOutputsBench(2);
	return 0;
}
