#ifndef __MTK_HMMA_F32_F32_DETAIL_HPP__
#define __MTK_HMMA_F32_F32_DETAIL_HPP__
#include <mma.h>
#include <type_traits>
#include <cuda_fp16.h>
#include "wmma_extension_include.hpp"

namespace mtk {
namespace wmma {
namespace mma_f32 {
namespace detail {
template <class Use, int a, int b, int c>
__device__ constexpr int select_value() {
	if constexpr (std::is_same<Use, nvcuda::wmma::matrix_a>::value) {
		return a;
	} else if constexpr (std::is_same<Use, nvcuda::wmma::matrix_b>::value) {
		return b;
	}
	return c;
}
template <class T>
__device__ constexpr int get_fragment_k() {return 16;};
template <> __device__ constexpr int get_fragment_k<nvcuda::wmma::precision::tf32>() {return 8 ;}

template <int frag_m, int frag_n, class Layout>
__device__ unsigned compute_mem_offset(const unsigned mem_offset, const unsigned ldm, const unsigned m_offset, const unsigned n_offset) {
	if constexpr (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		return (mem_offset % frag_m + m_offset) + ((mem_offset / frag_m) + n_offset) * ldm;
	}
	return ((mem_offset % frag_n) + n_offset) + (mem_offset / frag_n + m_offset) * ldm;
}

template <class Use, class T>
struct sub_frag_t {
	using type = T;
};
template <>
struct sub_frag_t<nvcuda::wmma::accumulator, half                         > {using type = float;};
template <>
struct sub_frag_t<nvcuda::wmma::accumulator, nvcuda::wmma::precision::tf32> {using type = float;};

template <class Layout, int a, int b>
__device__ constexpr int layout_switch() {
	if constexpr (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		return a;
	}
	return b;
}
} // detail
} // mma_f32
} // namespace wmma
} // namespace mtk
#endif
