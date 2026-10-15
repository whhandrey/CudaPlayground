#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"
#include "Benchmarks/3d/LocalMinBlendingBenchmark.h"

int main() {
	bench::Conv3dFusedSeparableMultipleOutputsBench(17);
	return 0;
}
