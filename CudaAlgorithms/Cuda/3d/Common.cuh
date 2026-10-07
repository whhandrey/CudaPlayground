#pragma once
#include <cuda_runtime.h>
#include <Image/ImageView.h>

__device__ __forceinline__ int FlatIdx(int x, int y, int z, int width, int height) {
	return x + y * width + z * width * height;
}

__device__ __forceinline__ size_t OffsetToRowPitched(int y, int z, size_t pitch, size_t slicePitch) {
	return y * pitch + z * slicePitch;
}

template <class T>
__device__ __forceinline__ T ReadPitched(const image::GpuVolumeView<const T>& view, int x, int y, int z) {
	size_t offsetBytes = OffsetToRowPitched(y, z, view.m_pitch, view.m_slicePitch);

	const T* row = (const T*)((const char*)view.m_ptr + offsetBytes);
	return row[x];
}

template <class T>
__device__ __forceinline__ T ReadPitched(const image::GpuVolumeView<T>& view, int x, int y, int z) {
	size_t offsetBytes = OffsetToRowPitched(y, z, view.m_pitch, view.m_slicePitch);

	const T* row = (const T*)((const char*)view.m_ptr + offsetBytes);
	return row[x];
}

template <class T>
__device__ __forceinline__ void WritePitched(image::GpuVolumeView<T>& view, T value, int x, int y, int z) {
	size_t offsetBytes = OffsetToRowPitched(y, z, view.m_pitch, view.m_slicePitch);

	T* row = (T*)((char*)view.m_ptr + offsetBytes);
	row[x] = value;
}
