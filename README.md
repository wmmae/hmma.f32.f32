# WMMA Extension for single precision matmul

An library for single precision matrix-matrix product using TensorCores and error correction technique

## Requirements
- CUDA
  - CUDA >= 10.0 for HMMA-FP16
  - CUDA >= 11.1 for HMMA-TF32

- C++ >= 17

## Sample code
```cuda
// sample.cu
#include <wmma_extension/hmma_f32_f32.hpp>

template <unsigned N>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
	__shared__ float smem[N * N];
	fill_zero(smem, N * N);

	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::col_major> frag_b;
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, half> frag_c, frag_d;

	// Load A
	// copy_matrix(smem, N, a_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_a, smem, N);

	// Load B
	// copy_matrix(smem, N, b_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_b, smem, N);

	// Load C
	// copy_matrix(smem, N, c_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_c, smem, N, nvcuda::wmma::mem_col_major);

	// Fill D
	mtk::wmma::fill_fragment(frag_d, 0.0f);

	// mma
	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	// Store D
	mtk::wmma::store_matrix_sync(smem, frag_d, N, nvcuda::wmma::mem_col_major);
	//copy_matrix(d_ptr, N, smem, N, N, N);
}
```
### Build
```bash
nvcc -I/path/to/hmma.f32.f32.f32/include/ -std=c++17 ...
```

## Fragment
```cpp
template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment_f32;
```
`mtk::wmma::fragment_f32` is a fragment for this computation.
It contains arrays of `nvcuda::wmma::fragment`.
- `m` and `n` has to be a multiple of 16.
- `k` has to be a multiple of 16 when `T` is `half` and 8 when `T` is `nvcuda::wmma::precision::tf32`.
- `T` is `half` or `nvcuda::wmma::precision::tf32`. Unlike `nvcuda::wmma::fragment`, even if `Use` is `nvcuda::wmma::accumulator`, the same is true.

## Functions
- `mtk::wmma::fill_fragment`
- `mtk::wmma::load_matrix_sync`
- `mtk::wmma::store_matrix_sync`
- `mtk::wmma::mma_sync`

- `mtk::wmma::load_vector`
- `mtk::wmma::store_vector`
- `mtk::wmma::fillzero`

## Lisence
MIT
