# WMMA Extension for single precision matmul

An extension library of WMMA API for single precision matrix operation using TensorCores and error correction technique

## Requirements
- CUDA
  - CUDA >= 10.0 for HMMA-FP16
  - CUDA >= 11.1 for HMMA-TF32

- C++ >= 17

## Sample code
```cuda
// sample.cu
// - Build
// nvcc -I/path/to/hmma.f32.f32.f32/include/ -std=c++17 sample.cu ...
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

## Fragment
```cpp
template <class Use, int m, int n, int k, class T, class Layout = void, Policy = typename mtk::wmma::detail::default_policy<T>::type>
struct fragment_f32;
```

### Template arguments
`mtk::wmma::fragment_f32` is a fragment for this computation.
It contains arrays of `nvcuda::wmma::fragment`.
- `m`, `n` and `k` have to be a multiple of `Policy::m`, `Policy::n` and `Policy::k` respectively.
You can get a default policy by `mtk::wmma::detail::default_policy<T>::type`.
- `k` has to be a multiple of 16 when `T` is `half` and 8 when `T` is `nvcuda::wmma::precision::tf32`.
- `T` is `half` or `nvcuda::wmma::precision::tf32`. Unlike `nvcuda::wmma::fragment`, even if `Use` is `nvcuda::wmma::accumulator`, the same is true.
- `Policy` is a concept of `mtk::wmma::detail::Policy<Op, ErrorCorrection, fm, fn, fk>`.
  - `Op` : `mtk::wmma::op_mma` / `mtk::wmma::op_wmma`
  - `ErrorCorrection` : `mtk::wmma::op_with_error_correction` / `mtk::wmma::op_without_error_correction`
  - `fm`, `fn`, `fk` is a size of internal fragments.

### Member variables/functions
- Member variable `element_type` is `float`
- Member function `x(index)` and `dx(index)` return the referrence of a elements.

## Functions
- `mtk::wmma::fill_fragment`
- `mtk::wmma::load_matrix_sync`
- `mtk::wmma::store_matrix_sync`
- `mtk::wmma::mma_sync`

- `mtk::wmma::load_vector`
- `mtk::wmma::store_vector`
- `mtk::wmma::fill_zero`

## Namespace
For easy portability, you can use `nvcuda` namespace instead of `mtk` by defining `WMMAE_USE_NVCUDA_NAMESPACE` before including header files.
```cpp
#define WMMA_USE_NVCUDA_NAMESPACE
#include <wmma_extension/hmma_f32_f32.hpp>

template <unsigned N>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
	__shared__ float smem[N * N];

	nvcuda::wmma::fragment_f32<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;

	// Load A
	// copy_matrix(smem, N, a_ptr, N, N, N);
	nvcuda::wmma::load_matrix_sync(frag_a, smem, N);

    // ...
}
```

## Lisence
MIT
