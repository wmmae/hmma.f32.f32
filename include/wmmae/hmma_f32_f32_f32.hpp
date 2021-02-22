#ifndef __MTK_HMMA_F32_F32_F32_HPP__
#define __MTK_HMMA_F32_F32_F32_HPP__
#include <type_traits>
#include "detail/wmma_extension/include/wmma_extension.hpp"

namespace mtk {
namespace wmma {
namespace detail {
template <class Use, int a, int b, int c>
constexpr int select_value() {
	if constexpr (std::is_same<Use, nvcuda::wmma::matrix_a>::value) {
		return a;
	} else if constexpr (std::is_same<Use, nvcuda::wmma::matrix_b>::value) {
		return b;
	}
	return c;
}
} // detail
template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment {
	static constexpr int sub_frag_m = detail::select_value<Use, m, k, m>() / 16;
	static constexpr int sub_frag_n = detail::select_value<Use, k, n, n>() / 16;
	nvcuda::wmma::fragment<Use, 16, 16, 16, half, Layout> sub_frag[sub_frag_m * sub_frag_n];
};
} // namespace wmma
} // namespace mtk
#endif
