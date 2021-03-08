#ifndef __MTK_HMMA_F32_F32_NO_COR_HPP__
#define __MTK_HMMA_F32_F32_NO_COR_HPP__

#include "detail/common.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace nvcuda {
#else
namespace mtk {
#endif
namespace wmma {
template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment_f32_no_cor {
	using element_type = T;

	using sub_frag_t = nvcuda::wmma::fragment<Use, 16, 16, mtk::wmma::detail::get_fragment_k<T>(), typename mtk::wmma::detail::sub_frag_t<Use, T>::type, Layout>;
	static constexpr int num_sub_frag_m = mtk::wmma::detail::select_value<Use, m, k, m>() / mtk::wmma::detail::select_value<Use, 16, mtk::wmma::detail::get_fragment_k<T>(), 16>();
	static constexpr int num_sub_frag_n = mtk::wmma::detail::select_value<Use, k, n, n>() / mtk::wmma::detail::select_value<Use, mtk::wmma::detail::get_fragment_k<T>(), 16, 16>();

	sub_frag_t sub_frag  [num_sub_frag_m * num_sub_frag_n];

	static const unsigned num_elements = num_sub_frag_m * num_sub_frag_m * sub_frag_t::num_elements;
	__device__ typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type& x(const unsigned index) {
		const auto frag_index = index % sub_frag_t::num_elements;
		const auto sub_frag_id = index / sub_frag_t::num_elements;
		return sub_frag[sub_frag_id].x[frag_index];
	}
};

template <class Use, int m, int n, int k, class T, class Layout = void>
__device__ void fill_fragment(fragment_f32_no_cor<Use, m, n, k, T, Layout>& frag, const typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type v) {
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			nvcuda::wmma::fill_fragment(frag.sub_frag  [bm + frag.num_sub_frag_m * bn], v);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout = void>
__device__ void fill_zero(fragment_f32_no_cor<Use, m, n, k, T, Layout>& frag) {
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::fill_zero(frag.sub_frag  [bm + frag.num_sub_frag_m * bn]);
		}
	}
}

// Load matrix
template <int m, int n, int k, class T>
__device__ void load_matrix_sync(fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag, const float* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
				nvcuda::wmma::load_matrix_sync(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
			}
		}
	} else {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
				nvcuda::wmma::load_matrix_sync(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
			}
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
__device__ void load_matrix_sync(fragment_f32_no_cor<Use, m, n, k, T, Layout>& frag, const float* const ptr, const unsigned ldm, const bool sync = true) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, 16, mtk::wmma::detail::get_fragment_k<T>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, mtk::wmma::detail::get_fragment_k<T>(), 16, 16>();

	mtk::wmma::foreach<decltype(frag.sub_frag[0])>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, Layout>(mem_index, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
						}
					}
				}
			});
	if (sync) {
		__syncthreads();
	}
}

// Store matrix
template <class Use, int m, int n, int k, class T>
__device__ void store_matrix_sync(float* const ptr, fragment_f32_no_cor<Use, m, n, k, T> frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			unsigned mem_offset;
			if (layout == nvcuda::wmma::mem_col_major) {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
			} else {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
			}
			nvcuda::wmma::store_matrix_sync(ptr + mem_offset, frag.sub_frag[bm + frag.num_sub_frag_m * bn], ldm, layout);
		}
	}
}

template <class Use, int m, int n, int k, class T>
__device__ void store_matrix_sync(float* const ptr, fragment_f32_no_cor<Use, m, n, k, T> frag, const unsigned ldm, const float mul, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			for (unsigned frag_index = 0; frag_index < frag.sub_frag[0].num_elements; frag_index++) {
				frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] * mul;
			}
			unsigned mem_offset;
			if (layout == nvcuda::wmma::mem_col_major) {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
			} else {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
			}
			nvcuda::wmma::store_matrix_sync(ptr + mem_offset, frag.sub_frag[bm + frag.num_sub_frag_m * bn], ldm, layout);
		}
	}
}

// Load vector
template <int m, int n, int k, class T>
__device__ void load_vector(fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag, const float* const ptr, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			mtk::wmma::load_vector(frag.sub_frag[bm], ptr + bm * frag_m, layout);
		}
	} else {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::load_vector(frag.sub_frag[bn * frag.num_sub_frag_m], ptr + bn * frag_n, layout);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
__device__ void load_vector(fragment_f32_no_cor<Use, m, n, k, T, Layout>& frag, const float* const ptr) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, 16, mtk::wmma::detail::get_fragment_k<T>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, mtk::wmma::detail::get_fragment_k<T>(), 16, 16>();

	constexpr auto num_load_blocks = mtk::wmma::detail::layout_switch<Layout, frag.num_sub_frag_m, frag.num_sub_frag_n>();
	constexpr auto block_ld        = mtk::wmma::detail::layout_switch<Layout, 1, frag.num_sub_frag_m>();
	constexpr auto vec_per_block   = mtk::wmma::detail::layout_switch<Layout, frag_m, frag_n>();

	mtk::wmma::foreach_v<decltype(frag.sub_frag[0])>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bn = 0; bn < num_load_blocks; bn++) {
					const auto mem_offset = mem_index + bn * vec_per_block;
					const auto v = ptr[mem_offset];
					const auto hv = mtk::wmma::detail::common::cast<T>(v);
					for (unsigned i = 0; i < frag_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag.sub_frag  [bn * block_ld].x[frag_index] = hv ;
					}
				}
			});
}

// Store vector
template <int m, int n, int k, class T>
__device__ void store_vector(float* const ptr, fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::foreach_v<decltype(frag.sub_frag[0])>(layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = hv;
						}
					}
				});
	} else {
		mtk::wmma::foreach_v<decltype(frag.sub_frag[0])>(layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = hv;
						}
					}
				});
	}
}

template <int m, int n, int k, class T>
__device__ void store_vector(float* const ptr, fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag, const float mul, const nvcuda::wmma::layout_t layout) {
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, 16, mtk::wmma::detail::get_fragment_k<float>(), 16>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, mtk::wmma::detail::get_fragment_k<float>(), 16, 16>();

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::foreach_v<decltype(frag.sub_frag[0])>(layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = hv * mul;
						}
					}
				});
	} else {
		mtk::wmma::foreach_v<decltype(frag.sub_frag[0])>(layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = hv * mul;
						}
					}
				});
	}
}

// mma
template <int m, int n, int k, class A_Layout, class B_Layout, class T>
__device__ void mma_sync(
		fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag_d,
		const fragment_f32_no_cor<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout>& frag_a,
		const fragment_f32_no_cor<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout>& frag_b,
		const fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag_c) {
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
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				nvcuda::wmma::mma_sync(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T>
__device__ void mma_sync(
		fragment_f32_no_cor<nvcuda::wmma::accumulator, m, n, k, T>& frag_d,
		const fragment_f32_no_cor<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout>& frag_a,
		const fragment_f32_no_cor<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout>& frag_b) {
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
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				nvcuda::wmma::mma_sync(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}
} // namespace wmma
} // namespace mtk
#endif
