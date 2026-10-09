#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"
#include "Benchmarks/3d/LocalMinBlendingBenchmark.h"

int main() {
	bench::localmin_blending3d::LocalMin3dFusedHaloStreamZBenchmark(5);
	return 0;
}
