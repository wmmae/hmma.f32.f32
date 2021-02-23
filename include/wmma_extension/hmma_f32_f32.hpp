#ifndef __MTK_HMMA_F32_F32_HPP__
#define __MTK_HMMA_F32_F32_HPP__
#include <type_traits>
#include <cuda_fp16.h>
#include "detail/wmma_extension/include/wmma_extension/wmma_extension.hpp"

namespace mtk {
namespace wmma {
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
} // detail

template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment_f32 {
	static constexpr int num_sub_frag_m = detail::select_value<Use, m, k, m>() / detail::select_value<Use, 16, detail::get_fragment_k<T>(), 16>();
	static constexpr int num_sub_frag_n = detail::select_value<Use, k, n, n>() / detail::select_value<Use, detail::get_fragment_k<T>(), 16, 16>();

	nvcuda::wmma::fragment<Use, 16, 16, detail::get_fragment_k<T>(), typename detail::sub_frag_t<Use, T>::type, Layout> sub_frag  [num_sub_frag_m * num_sub_frag_n];
	nvcuda::wmma::fragment<Use, 16, 16, detail::get_fragment_k<T>(), typename detail::sub_frag_t<Use, T>::type, Layout> sub_d_frag[num_sub_frag_m * num_sub_frag_n];
};

template <class Use, int m, int n, int k, class T, class Layout = void>
__device__ void fill_fragment(mtk::wmma::fragment_f32<Use, m, n, k, T, Layout>& frag, const typename mtk::wmma::detail::common::storage_t<typename detail::sub_frag_t<Use, T>::type>::type v) {
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			nvcuda::wmma::fill_fragment(frag.sub_frag  [bm + frag.num_sub_frag_m * bn], v);
			nvcuda::wmma::fill_fragment(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn], 0);
		}
	}
}

template <int m, int n, int k, class T>
__device__ void load_matrix_sync(mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T>& frag, const float* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = detail::select_value<nvcuda::wmma::accumulator, 16, detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = detail::select_value<nvcuda::wmma::accumulator, detail::get_fragment_k<float>(), 16, 16>();

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
				nvcuda::wmma::load_matrix_sync(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
				mtk::wmma::fill_zero(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn]);
			}
		}
	} else {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
				nvcuda::wmma::load_matrix_sync(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
				mtk::wmma::fill_zero(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn]);
			}
		}
	}
}

template <class Use, int m, int n, int k, class Layout>
__device__ void load_matrix_sync(mtk::wmma::fragment_f32<Use, m, n, k, half, Layout>& frag, const float* const ptr, const unsigned ldm) {
	constexpr auto frag_m = detail::select_value<Use, 16, detail::get_fragment_k<half>(), 16>();
	constexpr auto frag_n = detail::select_value<Use, detail::get_fragment_k<half>(), 16, 16>();

	mtk::wmma::foreach<decltype(frag.sub_frag[0])>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = detail::compute_mem_offset<frag_m, frag_n, Layout>(mem_index, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<half>(v);
						const auto dhv = mtk::wmma::detail::common::cast<half>((v - mtk::wmma::detail::common::cast<float>(hv)) * 1024);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
						}
					}
				}
			});
}
template <class Use, int m, int n, int k, class Layout>
__device__ void load_matrix_sync(mtk::wmma::fragment_f32<Use, m, n, k, nvcuda::wmma::precision::tf32, Layout>& frag, const float* const ptr, const unsigned ldm) {
	constexpr auto frag_m = detail::select_value<Use, 16, detail::get_fragment_k<nvcuda::wmma::precision::tf32>(), 16>();
	constexpr auto frag_n = detail::select_value<Use, detail::get_fragment_k<nvcuda::wmma::precision::tf32>(), 16, 16>();

	mtk::wmma::foreach<decltype(frag.sub_frag[0])>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = detail::compute_mem_offset<frag_m, frag_n, Layout>(mem_index, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(v);
						const auto dhv = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>((v - mtk::wmma::detail::common::cast<float>(hv)));
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
						}
					}
				}
			});
}

template <class Use, int m, int n, int k, class T>
__device__ void store_matrix_sync(float* const ptr, mtk::wmma::fragment_f32<Use, m, n, k, T> frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = detail::select_value<Use, 16, detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = detail::select_value<Use, detail::get_fragment_k<float>(), 16, 16>();

	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			if constexpr (std::is_same<T, half>::value) {
				for (unsigned frag_index = 0; frag_index < frag.sub_frag[0].num_elements; frag_index++) {
					frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] += frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] / 1024;
				}
			} else {
				for (unsigned frag_index = 0; frag_index < frag.sub_frag[0].num_elements; frag_index++) {
					frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] += frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index];
				}
			}
			unsigned mem_offset;
			if (layout == nvcuda::wmma::mem_col_major) {
				mem_offset = detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
			} else {
				mem_offset = detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
			}
			nvcuda::wmma::store_matrix_sync(ptr + mem_offset, frag.sub_frag[bm + frag.num_sub_frag_m * bn], ldm, layout);
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T>
__device__ void mma_sync(
		mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T>& frag_d,
		const mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout>& frag_a,
		const mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout>& frag_b,
		const mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T>& frag_c) {
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			nvcuda::wmma::mma_sync(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_c.sub_frag[bm + bn * num_m_block]
					);
			nvcuda::wmma::mma_sync(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_d_frag[bm + 0  * num_m_block],
					frag_b.sub_frag  [0  + bn * num_k_block],
					frag_c.sub_d_frag[bm + bn * num_m_block]
					);
			nvcuda::wmma::mma_sync(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_frag  [bm + 0  * num_m_block],
					frag_b.sub_d_frag[0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				nvcuda::wmma::mma_sync(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
				nvcuda::wmma::mma_sync(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_d_frag[bm + bk * num_m_block],
						frag_b.sub_frag  [bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
				nvcuda::wmma::mma_sync(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_frag  [bm + bk * num_m_block],
						frag_b.sub_d_frag[bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T>
__device__ void mma_sync(
		mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T>& frag_d,
		const mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout>& frag_a,
		const mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout>& frag_b) {
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			mtk::wmma::fill_zero(frag_d.sub_frag[bm + bn * num_m_block]);
			nvcuda::wmma::mma_sync(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_d.sub_frag[bm + bn * num_m_block]
					);
			mtk::wmma::fill_zero(frag_d.sub_d_frag[bm + bn * num_m_block]);
			nvcuda::wmma::mma_sync(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_d_frag[bm + 0  * num_m_block],
					frag_b.sub_frag  [0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			nvcuda::wmma::mma_sync(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_frag  [bm + 0  * num_m_block],
					frag_b.sub_d_frag[0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				nvcuda::wmma::mma_sync(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
				nvcuda::wmma::mma_sync(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_d_frag[bm + bk * num_m_block],
						frag_b.sub_frag  [bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
				nvcuda::wmma::mma_sync(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_frag  [bm + bk * num_m_block],
						frag_b.sub_d_frag[bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}
} // namespace wmma
} // namespace mtk
#endif
