#ifndef __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_HPP__
#include "wmma_extension/include/wmma_extension/wmma_mma.hpp"
#include "wmma_extension/include/wmma_extension/wmma_extension.hpp"
#include "policy.hpp"
namespace mtk {
namespace wmma {
namespace detail {
// foreach
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct foreach_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class Func>
struct foreach_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>, Func> {
	void operator()(Func func) {
		mtk::wmma::foreach<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				func
				);
	}
	void operator()(Func func, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::foreach<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				layout, func
				);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class Func>
struct foreach_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>, Func> {
	void operator()(Func func) {
		mtk::wmma::mma::foreach<typename mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>>(
				func
				);
	}
	void operator()(Func func, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::foreach<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				layout, func
				);
	}
};

// foreach_v
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct foreach_v_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class Func>
struct foreach_v_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>, Func> {
	void operator()(Func func) {
		mtk::wmma::foreach_v<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				func
				);
	}
	void operator()(Func func, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::foreach_v<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				layout, func
				);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class Func>
struct foreach_v_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>, Func> {
	void operator()(Func func) {
		mtk::wmma::mma::foreach_v<typename mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>>(
				func
				);
	}
	void operator()(Func func, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::foreach_v<typename nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>(
				layout, func
				);
	}
};

// fill zero
template <class Use, int m, int n, int k, class T, class Layout, class Policy>
struct fill_zero_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct fill_zero_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>{
	void operator()(nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag) {
		mtk::wmma::fill_zero(frag);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct fill_zero_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>> {
	void operator()(mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag) {
		mtk::wmma::mma::fill_zero(frag);
	}
};

// load_matrix_sync
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct load_matrix_sync_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct load_matrix_sync_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>> {
	void operator()(nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, float* ptr, unsigned ldm, const nvcuda::wmma::layout_t layout) {
		nvcuda::wmma::load_matrix_sync(frag, ptr, ldm, layout);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct load_matrix_sync_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>> {
	void operator()(mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag, float* ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::load_matrix_sync(frag, ptr, ldm, layout);
	}
};

// store_matrix_sync
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct store_matrix_sync_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct store_matrix_sync_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>> {
	void operator()(float* ptr, nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, unsigned ldm, const nvcuda::wmma::layout_t layout) {
		nvcuda::wmma::store_matrix_sync(ptr, frag, ldm, layout);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct store_matrix_sync_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>> {
	void operator()(float* ptr, mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::store_matrix_sync(ptr, frag, ldm, layout);
	}
};

// load_vector
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct load_vector_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct load_vector_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>> {
	void operator()(nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, float* ptr, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::load_vector(frag, ptr, ldm, layout);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct load_vector_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>> {
	void operator()(mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag, float* ptr, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::load_vector(frag, ptr, layout);
	}
};

// store_vector
template <class Use, int m, int n, int k, class T, class Layout, class Policy, class Func>
struct store_vector_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct store_vector_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>> {
	void operator()(float* ptr, nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::store_vector(ptr, frag, layout);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk>
struct store_vector_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>> {
	void operator()(float* ptr, mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::store_vector_wrapper(ptr, frag, layout);
	}
};

// fill_fragment
template <class Use, int m, int n, int k, class T, class Layout, class Policy>
struct fill_fragment_wrapper;

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class VT>
struct fill_fragment_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_wmma, fm, fn, fk>, VT> {
	void operator()(nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, const VT v) {
		nvcuda::wmma::fill_fragment(frag, v);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, int fm, int fn, int fk, class VT>
struct fill_fragment_wrapper<Use, m, n, k, T, Layout, Policy<mtk::wmma::op_mma, fm, fn, fk>, VT> {
	void operator()(mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>& frag, const VT v) {
		mtk::wmma::mma::fill_fragment(frag, v);
	}
};

} // namespace detail
} // namespace wmma
} // namespace mtk
#endif
