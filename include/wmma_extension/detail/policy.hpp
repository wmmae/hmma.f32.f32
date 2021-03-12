#ifndef __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#include <mma.h>
#include "wmma_extension/include/wmma_extension/wmma_mma.hpp"
namespace mtk {
namespace wmma {

struct op_mma;
struct op_wmma;

namespace detail {
template <class op_, int m_, int n_, int k_>
struct Policy {
	using op = op_;
	static const int m = m_;
	static const int n = n_;
	static const int k = k_;
};

// ===================================
// Default policy selector
// ===================================
template <class T>
struct default_policy;
template <>
struct default_policy<half                         > {using type = mtk::wmma::detail::Policy<mtk::wmma::op_wmma, 16, 16, 16>;};
template <>
struct default_policy<nvcuda::wmma::precision::tf32> {using type = mtk::wmma::detail::Policy<mtk::wmma::op_wmma, 16, 16, 8 >;};


// ===================================
// Default fragment selector
// ===================================
template <class Use, int m, int n, int k, class T, class Layout, class Policy>
struct default_fragment;

template <class Use, int m, int n, int k, class T, class Layout>
struct default_fragment<Use, m, n, k, T, Layout, Policy<op_wmma, 16, 16, 16>> {
	using type = nvcuda::wmma::fragment<Use, 16, 16, 16, T, Layout>;
};
template <class Use, int m, int n, int k, class T, class Layout>
struct default_fragment<Use, m, n, k, T, Layout, Policy<op_wmma, 16, 16, 8 >> {
	using type = nvcuda::wmma::fragment<Use, 16, 16, 8, T, Layout>;
};

template <class Use, int m, int n, int k, class T, class Layout>
struct default_fragment<Use, m, n, k, T, Layout, Policy<op_mma, 16, 8, 16>> {
	using type = mtk::wmma::mma::fragment<Use, 16, 8, 16, T, Layout>;
};
} // namespace detail

} // namespace wmma
} // namespace mtk
#endif
