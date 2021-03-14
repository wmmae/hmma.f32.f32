#ifndef __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#include <mma.h>
#include "wmma_extension/include/wmma_extension/wmma_mma.hpp"
namespace mtk {
namespace wmma {

// Instruction policy
struct op_mma;
struct op_wmma;

// Error correction policy
struct op_with_error_correction;
struct op_without_error_correction;

namespace detail {
template <class op_, class ec_, int m_, int n_, int k_>
struct Policy {
	using op = op_;
	using error_correction = ec_;
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
struct default_policy<half                         > {using type = mtk::wmma::detail::Policy<mtk::wmma::op_wmma, mtk::wmma::op_with_error_correction, 16, 16, 16>;};
template <>
struct default_policy<nvcuda::wmma::precision::tf32> {using type = mtk::wmma::detail::Policy<mtk::wmma::op_wmma, mtk::wmma::op_with_error_correction, 16, 16, 8 >;};


// ===================================
// Default fragment selector
// ===================================
template <class Use, class T, class Layout, class Policy>
struct default_fragment;

template <class Use, class T, class Layout, class ec, int fm, int fn, int fk>
struct default_fragment<Use, T, Layout, Policy<op_wmma, ec, fm, fn, fk>> {
	using type = nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>;
};
} // namespace detail

} // namespace wmma
} // namespace mtk
#endif
