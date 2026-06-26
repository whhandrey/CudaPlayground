#pragma once

#include <Image/ImageTypes.h>
#include <cuda_runtime.h>

struct ALIGN(8) range {
	int min;
	int max;
};

struct SadCandidate {
	int sad;
	int dx;
	int dy;
};

__device__ __forceinline__ SadCandidate MinSad(SadCandidate first, SadCandidate second) {
	return first.sad < second.sad ? first : second;
}
