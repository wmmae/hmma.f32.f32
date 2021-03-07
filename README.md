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
template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment_f32;
```
`mtk::wmma::fragment_f32` is a fragment for this computation.
It contains arrays of `nvcuda::wmma::fragment`.
- `m`, `n` and `k` have to be a multiple of `mtk::wmma::min_fragment_m<T>`, `mtk::wmma::min_fragmnet_n<T>` and `mtk::wmma::min_fragmnet_k<T>` respectively.
Currently `mtk::wmma::min_fragment_m<T>` and `mtk::wmma::min_fragmnet_n<T>` are 16 and `mtk::wmma::min_fragmnet_n<T>` is 16 for `T` == `half` and 8 for `T` == `nvcuda::wmma::precision::tf32`.
- `k` has to be a multiple of 16 when `T` is `half` and 8 when `T` is `nvcuda::wmma::precision::tf32`.
- `T` is `half` or `nvcuda::wmma::precision::tf32`. Unlike `nvcuda::wmma::fragment`, even if `Use` is `nvcuda::wmma::accumulator`, the same is true.

## Functions
- `mtk::wmma::fill_fragment`
- `mtk::wmma::load_matrix_sync`
- `mtk::wmma::store_matrix_sync`
- `mtk::wmma::mma_sync`

- `mtk::wmma::load_vector`
- `mtk::wmma::store_vector`
- `mtk::wmma::fill_zero`

## Evaluation of the effect of this correction technique
To evaluate the effect of this correction technique, this library also provides no correcting fragment.
To make it easy to use it, you can define and use a helper fragment selector like this.

```cpp
#include <wmma_extension/hmma_f32_f32.hpp>
#include <wmma_extension/hmma_f32_f32_no_cor.hpp>

template <bool Cor, class Use, unsigned m, unsigned n, unsigned k, class T, class Layout = void>
struct select_fragemnt {
	using type = void;
};

template <class Use, unsigned m, unsigned n, unsigned k, class T, class Layout>
struct select_fragemnt<true , Use, m, n, k, T, Layout> {
	using type = typename mtk::wmma::fragment_f32<Use, m, n, k, T, Layout>;
};

template <class Use, unsigned m, unsigned n, unsigned k, class T, class Layout>
struct select_fragemnt<false, Use, m, n, k, T, Layout> {
	using type = typename mtk::wmma::fragment_f32_no_cor<Use, m, n, k, T, Layout>;
};
```

Then use like this.

```cpp
template <bool Cor>
void kernel() {
	typename select_fragemnt<Cor, nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major>::type frag_a;
	typename select_fragemnt<Cor, nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major>::type frag_b;
	typename select_fragemnt<Cor, nvcuda::wmma::accumulator, N, N, N, T>::type frag_c, frag_d;
	// ...
}
```

## Namespace
For easy portability, you can use `nvcuda` namespace instead of `mtk` by defining `WMMAE_USE_NVCUDA_NAMESPACE` before including header files.
```cpp
#define WMMA_USE_NVCUDA_NAMESPACE
#include <wmma_extension/hmma_f32_f32.hpp>

// ...
```

## Lisence
MIT
