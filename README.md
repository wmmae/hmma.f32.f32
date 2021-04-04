# WMMA Extension for single precision matmul

An extension library of WMMA API for single precision matrix operation using TensorCores and error correction technique

## Correction tequnique
See this slide page 9 [slide](https://static.momo86.net/f/1/cse21-slide)  
Hiroyuki Ootomo, Rio Yokota. TSQR on TensorCores with error correction. SIAM CSE'21

## Requirements
- CUDA
  - CUDA >= 10.0 for HMMA-FP16
  - CUDA >= 11.1 for HMMA-TF32

- C++ >= 17

## Installation and build
This library dependes on [wmma_extension](https://github.com/wmmae/wmma_extension) library.

1. Clone [wmma_extension](https://github.com/wmmae/wmma_extension) and [mma.f32.f32](https://github.com/wmmae/mma.f32.f32).
```bash
git clone https://github.com/wmmae/wmma_extension
git clone https://github.com/wmmae/mma.f32.f32
```

2. Build
```bash
nvcc -I/path/to/hmma.f32.f32/include/ -I./path/to/wmma_extension/include/ -std=c++17 sample.cu ...
```

When you can't set `-I` options, include headers like blow.
```cuda
#include "path/to/wmma_extension/include/wmma_extension/wmma_extension.hpp"
#include "path/to/wmma_extension/include/wmma_extension/wmma_mma.hpp"
#define WMMAE_NOT_INCLUDE_WMMAE_HEADER
#include "path/to/mma.f32.f32/include/wmma_extension/hmma_f32_f32.hpp"
```

## Sample code
```cuda
// sample.cu
//
#include <wmma_extension/hmma_f32_f32.hpp>

template <unsigned N>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
    __shared__ float smem[N * N];
    fill_zero(smem, N * N);

    mtk::wmma::mma_f32::fragment<nvcuda::wmma::mma_f32::matrix_a, N, N, N, half, nvcuda::wmma::mma_f32::col_major> frag_a;
    mtk::wmma::mma_f32::fragment<nvcuda::wmma::mma_f32::matrix_b, N, N, N, half, nvcuda::wmma::mma_f32::col_major> frag_b;
    mtk::wmma::mma_f32::fragment<nvcuda::wmma::mma_f32::accumulator, N, N, N, half> frag_c, frag_d;

    // Load A
    // copy_matrix(smem, N, a_ptr, N, N, N);
    mtk::wmma::mma_f32::load_matrix_sync(frag_a, smem, N);

    // Load B
    // copy_matrix(smem, N, b_ptr, N, N, N);
    mtk::wmma::mma_f32::load_matrix_sync(frag_b, smem, N);

    // Load C
    // copy_matrix(smem, N, c_ptr, N, N, N);
    mtk::wmma::mma_f32::load_matrix_sync(frag_c, smem, N, nvcuda::wmma::mma_f32::mem_col_major);

    // Fill D
    mtk::wmma::mma_f32::fill_fragment(frag_d, 0.0f);

    // mma
    mtk::wmma::mma_f32::mma_sync(frag_d, frag_a, frag_b, frag_c);

    // Store D
    mtk::wmma::mma_f32::store_matrix_sync(smem, frag_d, N, nvcuda::wmma::mma_f32::mem_col_major);
    //copy_matrix(d_ptr, N, smem, N, N, N);
}
```

## Fragment
```cpp
template <class Use, int m, int n, int k, class T, class Layout = void, Policy = typename mtk::wmma::mma_f32::detail::default_policy<T>::type>
struct fragment;
```

### Template arguments
`mtk::wmma::mma_f32::fragment` is a fragment for this computation.
It contains arrays of `nvcuda::wmma::fragment`.
- `m`, `n` and `k` have to be a multiple of `Policy::m`, `Policy::n` and `Policy::k` respectively.
You can get a default policy by `mtk::wmma::mma_f32::detail::default_policy<T>::type`.
- `k` has to be a multiple of 16 when `T` is `half` and 8 when `T` is `nvcuda::wmma::precision::tf32`.
- `T` is `half` or `nvcuda::wmma::precision::tf32`. Unlike `nvcuda::wmma::fragment`, even if `Use` is `nvcuda::wmma::accumulator`, the same is true.
- `Policy` is a concept of `mtk::wmma::mma_f32::Policy<Op, ErrorCorrection, fm, fn, fk>`.
  - `Op` : `mtk::wmma::mma_f32::op_mma` / `mtk::wmma::mma_f32::op_wmma`
  - `ErrorCorrection` : `mtk::wmma::mma_f32::op_with_error_correction` / `mtk::wmma::mma_f32::op_without_error_correction`
  - `fm`, `fn`, `fk` is a size of internal fragments.

## Supported fragment

| fm | fn | fk | LayoutA | LayoutB | Type |
| -- | -- | -- | ------- | ------- | ---- |
| 16 | 16 | 16 | col/row | col/row | half |
| 16 | 16 | 16 | col/row | col/row | tf32 |
| 16 | 8  | 16 | row     | col     | half |
| 16 | 8  | 8  | row     | col     | half |

### Member variables/functions
- Member variable `element_type` is `float`
- Member function `x(index)` and `dx(index)` return the referrence of a elements.

## Functions
- `mtk::wmma::mma_f32::fill_fragment`
- `mtk::wmma::mma_f32::load_matrix_sync`
- `mtk::wmma::mma_f32::store_matrix_sync`
- `mtk::wmma::mma_f32::mma_sync`

- `mtk::wmma::mma_f32::load_vector`
- `mtk::wmma::mma_f32::store_vector`
- `mtk::wmma::mma_f32::fill_zero`

## Lisence
MIT
