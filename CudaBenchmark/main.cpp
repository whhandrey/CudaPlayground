#include "Benchmarks/3d/Conv3dBenchmark.h"
#include "Benchmarks/3d/Sobel3dBenchmark.h"
#include "Benchmarks/3d/LocalMinBlendingBenchmark.h"
#include "Benchmarks/3d/SynthVolumePipelineBench.h"

int main() {
	bench::synth_pipeline::SynthVolPipelineOptBench();
	return 0;
}
