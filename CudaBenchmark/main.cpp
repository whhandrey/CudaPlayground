#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"

int main() {
	bench::Conv3dCheckIfMatchWithNaive(bench::Conv3dType::FusedSeparableMultipleOutputs);
	return 0;
}
