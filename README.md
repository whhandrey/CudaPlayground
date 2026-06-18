# CudaPlayground

Project for experimenting with different cuda stuff and visualizing the results.

## Third-party libraries

### stb

Vendored from https://github.com/nothings/stb

Used files:
- stb_image.h
- stb_image_write.h

stb is public domain / MIT-style licensed. See `ThirdParty/stb/LICENSE`.

## Benchmark notes

On hybrid GPU laptops, CUDA benchmark results depend on GPU power state.

For stable RTX 4050 Laptop results:
- Add `CudaBenchmark.exe` to Windows Graphics settings.
- Set it to High performance / NVIDIA GPU.
- Run one warmup pass before trusting numbers.
- Reject results if RTX 4050 SM clock stays around ~780 MHz instead of boosting above ~2000 MHz.

Known clean baseline:
- 1280x960
- macroblock 16x16
- search halfsize 3x3
- `BlockMatchingSimpleKernel (8,8)` median ≈ 0.35 ms
- `BlockMatchingSimpleKernel_T (8,8)` median ≈ 0.30 ms
