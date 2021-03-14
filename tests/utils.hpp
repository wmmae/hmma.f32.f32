#ifndef __HMMA_F32_F32_TEST_UTILS_HPP__
#define __HMMA_F32_F32_TEST_UTILS_HPP__
#include <cuda_fp16.h>
#include <string>
#include <wmma_extension/hmma_f32_f32.hpp>

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace fragment_f32_namespace = nvcuda;
#else
namespace fragment_f32_namespace = mtk;
#endif

namespace mtk {
namespace test_utils {

template <class T>
std::string to_string();
template <> std::string to_string<nvcuda::wmma::accumulator>    (){return "acc";}
template <> std::string to_string<nvcuda::wmma::matrix_a>       (){return "matrix_a";}
template <> std::string to_string<nvcuda::wmma::matrix_b>       (){return "matrix_b";}
template <> std::string to_string<nvcuda::wmma::col_major>      (){return "col_major";}
template <> std::string to_string<nvcuda::wmma::row_major>      (){return "row_major";}
template <> std::string to_string<float>                        (){return "float";}
template <> std::string to_string<half>                         (){return "half";}
template <> std::string to_string<nvcuda::wmma::precision::tf32>(){return "tf32";}

template <bool Cor, class Use, unsigned m, unsigned n, unsigned k, class T, class Layout = void>
struct select_fragemnt {
	using type = void;
};

template <class Use, unsigned m, unsigned n, unsigned k, class T, class Layout>
struct select_fragemnt<true , Use, m, n, k, T, Layout> {
	using type = typename fragment_f32_namespace::wmma::fragment_f32<Use, m, n, k, T, Layout, typename mtk::wmma::detail::default_policy<T, mtk::wmma::op_with_error_correction>::type>;
};

template <class Use, unsigned m, unsigned n, unsigned k, class T, class Layout>
struct select_fragemnt<false, Use, m, n, k, T, Layout> {
	using type = typename fragment_f32_namespace::wmma::fragment_f32<Use, m, n, k, T, Layout, typename mtk::wmma::detail::default_policy<T, mtk::wmma::op_without_error_correction>::type>;
};


constexpr unsigned warp_size = 32;

__device__ void copy_matrix(
		float* const dst, const unsigned ldd,
		const float* const src, const unsigned lds,
		const unsigned m, const unsigned n) {
	for (unsigned i = 0; i < m * n; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= m * n) return;
		const auto mm = j % m;
		const auto mn = j / m;
		dst[mm + mn * ldd] = src[mm + mn * lds];
	}
}

__device__ void fill_zero(float* const dst, const unsigned size) {
	for (unsigned i = 0; i < size; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= size) return;
		dst[j] = 0.0f;
	}
}
}
}
#endif
