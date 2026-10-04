#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"

int main() {
	bench::SobelMag3dNaiveSharedMemBench();
	return 0;
}
