#ifndef __MTK_HMMA_F32_F32_HPP__
#define __MTK_HMMA_F32_F32_HPP__
#include "detail/common.hpp"
#include "detail/policy.hpp"
#include "detail/functions.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace nvcuda {
#else
namespace mtk {
#endif
namespace wmma {

namespace detail {
template <class T>
__device__ float correction_scale_0(const float v) {return v;}
template <>
__device__ float correction_scale_0<half>(const float v) {return v * 1024;}

template <class T>
__device__ float correction_scale_1(const float v) {return v;}
template <>
__device__ float correction_scale_1<half>(const float v) {return v / 1024;}
}

template <class Use, int m, int n, int k, class T, class Layout = void,
		 class Policy_ = typename mtk::wmma::detail::default_policy<T>::type>
struct fragment_f32 {
	using Policy = Policy_;
	using element_type = float;
	static const int sub_frag_m = Policy::m;
	static const int sub_frag_n = Policy::n;
	static const int sub_frag_k = Policy::k;

	using sub_frag_t = typename mtk::wmma::detail::default_fragment<Use, typename mtk::wmma::detail::sub_frag_t<Use, T>::type, Layout, Policy>::type;
	static constexpr int num_sub_frag_m = mtk::wmma::detail::select_value<Use, m, k, m>() / mtk::wmma::detail::select_value<Use, sub_frag_m, sub_frag_k, sub_frag_m>();
	static constexpr int num_sub_frag_n = mtk::wmma::detail::select_value<Use, k, n, n>() / mtk::wmma::detail::select_value<Use, sub_frag_k, sub_frag_n, sub_frag_n>();

	sub_frag_t sub_frag  [num_sub_frag_m * num_sub_frag_n];
	sub_frag_t sub_d_frag[num_sub_frag_m * num_sub_frag_n];

	static const unsigned num_elements = num_sub_frag_m * num_sub_frag_n * sub_frag_t::num_elements;
	__device__ typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type& x(const unsigned index) {
		const auto frag_index = index % sub_frag_t::num_elements;
		const auto sub_frag_id = index / sub_frag_t::num_elements;
		return sub_frag[sub_frag_id].x[frag_index];
	}
	__device__ typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type& dx(const unsigned index) {
		const auto frag_index = index % sub_frag_t::num_elements;
		const auto sub_frag_id = index / sub_frag_t::num_elements;
		return sub_d_frag[sub_frag_id].x[frag_index];
	}
};

// min m
template <class Frag>
struct min_fragment_m;
template <class Use, int m, int n, int k, class T, class Layout, class Op, class ErrorCorrection, int fn, int fm, int fk>
struct min_fragment_m<fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, ErrorCorrection, fm, fn, fk>>> {static const unsigned value = fm;};
// min n
template <class Frag>
struct min_fragment_n;
template <class Use, int m, int n, int k, class T, class Layout, class Op, class ErrorCorrection, int fn, int fm, int fk>
struct min_fragment_n<fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, ErrorCorrection, fm, fn, fk>>> {static const unsigned value = fn;};
// min k
template <class Frag>
struct min_fragment_k;
template <class Use, int m, int n, int k, class T, class Layout, class Op, class ErrorCorrection, int fn, int fm, int fk>
struct min_fragment_k<fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, ErrorCorrection, fm, fn, fk>>> {static const unsigned value = fk;};

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void fill_fragment(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type v) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	using sub_frag_t = typename mtk::wmma::detail::sub_frag_t<Use, T>::type;
	using storage_t = typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::detail::sub_frag_t<Use, T>::type>::type;
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::detail::fill_fragment_wrapper<Use, sub_frag_t, Layout, Policy, storage_t>{}(frag.sub_frag  [bm + frag.num_sub_frag_m * bn], v);
			mtk::wmma::detail::fill_fragment_wrapper<Use, sub_frag_t, Layout, Policy, storage_t>{}(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn], 0);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void fill_zero(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::detail::fill_zero_wrapper<Use, T, Layout, Policy>{}(frag.sub_frag  [bm + frag.num_sub_frag_m * bn]);
			mtk::wmma::detail::fill_zero_wrapper<Use, T, Layout, Policy>{}(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn]);
		}
	}
}

// Load matrix
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
				mtk::wmma::detail::load_matrix_sync_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
				mtk::wmma::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn]);
			}
		}
	} else {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
				auto mem_offset = 0u;
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
				mtk::wmma::detail::load_matrix_sync_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bm + frag.num_sub_frag_m * bn], ptr + mem_offset, ldm, layout);
				mtk::wmma::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn]);
			}
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, Policy::k, Policy::n, Policy::n>();

	mtk::wmma::detail::foreach_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, Layout>(mem_index, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
						}
					}
				}
			});
	if (sync) {
		__syncthreads();
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr, const unsigned ldm, const float mul, const bool sync = true) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, Policy::k, Policy::n, Policy::n>();

	mtk::wmma::detail::foreach_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, Layout>(mem_index, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset] * mul;
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
						}
					}
				}
			});
	if (sync) {
		__syncthreads();
	}
}

// Store matrix
// [Important!!]
// `frag` must not be a ref because this function breaks frag.
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(float* const ptr, fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>> frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			for (unsigned frag_index = 0; frag_index < frag.sub_frag[0].num_elements; frag_index++) {
				frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] += detail::correction_scale_1<T>(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index]);
			}
			unsigned mem_offset;
			if (layout == nvcuda::wmma::mem_col_major) {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
			} else {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
			}
			mtk::wmma::detail::store_matrix_sync_wrapper<T, Policy>{}(ptr + mem_offset, frag.sub_frag[bm + frag.num_sub_frag_m * bn], ldm, layout);
		}
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(float* const ptr, fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>> frag, const unsigned ldm, const float mul, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			for (unsigned frag_index = 0; frag_index < frag.sub_frag[0].num_elements; frag_index++) {
				frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = (frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] + detail::correction_scale_1<T>(frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index])) * mul;
			}
			unsigned mem_offset;
			if (layout == nvcuda::wmma::mem_col_major) {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>(0, ldm, bm * frag_m, bn * frag_n);
			} else {
				mem_offset = mtk::wmma::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>(0, ldm, bm * frag_m, bn * frag_n);
			}
			mtk::wmma::detail::store_matrix_sync_wrapper<T, Policy>{}(ptr + mem_offset, frag.sub_frag[bm + frag.num_sub_frag_m * bn], ldm, layout);
		}
	}
}

// Load vector
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void load_vector(fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			mtk::wmma::detail::load_vector_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bm], ptr + bm * frag_m, layout);
		}
	} else {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::detail::load_vector_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bn * frag.num_sub_frag_m], ptr + bn * frag_n, layout);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_vector(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, Policy::k, Policy::n, Policy::n>();

	constexpr auto num_load_blocks = mtk::wmma::detail::layout_switch<Layout, frag.num_sub_frag_m, frag.num_sub_frag_n>();
	constexpr auto block_ld        = mtk::wmma::detail::layout_switch<Layout, 1, frag.num_sub_frag_m>();
	constexpr auto vec_per_block   = mtk::wmma::detail::layout_switch<Layout, frag_m, frag_n>();

	mtk::wmma::detail::foreach_v_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bn = 0; bn < num_load_blocks; bn++) {
					const auto mem_offset = mem_index + bn * vec_per_block;
					const auto v = ptr[mem_offset];
					const auto hv = mtk::wmma::detail::common::cast<T>(v);
					const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
					for (unsigned i = 0; i < frag_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag.sub_frag  [bn * block_ld].x[frag_index] = hv ;
						frag.sub_d_frag[bn * block_ld].x[frag_index] = dhv;
					}
				}
			});
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_vector(fragment_f32<Use, m, n, k, T, Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float* const ptr, const float mul) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<Use, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<Use, Policy::k, Policy::n, Policy::n>();

	constexpr auto num_load_blocks = mtk::wmma::detail::layout_switch<Layout, frag.num_sub_frag_m, frag.num_sub_frag_n>();
	constexpr auto block_ld        = mtk::wmma::detail::layout_switch<Layout, 1, frag.num_sub_frag_m>();
	constexpr auto vec_per_block   = mtk::wmma::detail::layout_switch<Layout, frag_m, frag_n>();

	mtk::wmma::detail::foreach_v_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bn = 0; bn < num_load_blocks; bn++) {
					const auto mem_offset = mem_index + bn * vec_per_block;
					const auto v = ptr[mem_offset] * mul;
					const auto hv = mtk::wmma::detail::common::cast<T>(v);
					const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
					for (unsigned i = 0; i < frag_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag.sub_frag  [bn * block_ld].x[frag_index] = hv ;
						frag.sub_d_frag[bn * block_ld].x[frag_index] = dhv;
					}
				}
			});
}

// Store vector
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_vector(float* const ptr, fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							const auto dhv = frag.sub_d_frag[bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = (hv + detail::correction_scale_1<T>(dhv));
						}
					}
				});
	} else {
		mtk::wmma::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							const auto dhv = frag.sub_d_frag[bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = (hv + detail::correction_scale_1<T>(dhv));
						}
					}
				});
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_vector(float* const ptr, fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag, const float mul, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>();
	constexpr auto frag_n = mtk::wmma::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>();

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							const auto dhv = frag.sub_d_frag[bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = (hv + detail::correction_scale_1<T>(dhv)) * mul;
						}
					}
				});
	} else {
		mtk::wmma::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							const auto dhv = frag.sub_d_frag[bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = (hv + detail::correction_scale_1<T>(dhv)) * mul;
						}
					}
				});
	}
}

// mma
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk>
__device__ void mma_sync(
		fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_d,
		const fragment_f32<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_a,
		const fragment_f32<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_b,
		const fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_c) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			mma_op(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_c.sub_frag[bm + bn * num_m_block]
					);
			mma_op(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_d_frag[bm + 0  * num_m_block],
					frag_b.sub_frag  [0  + bn * num_k_block],
					frag_c.sub_d_frag[bm + bn * num_m_block]
					);
			mma_op(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_frag  [bm + 0  * num_m_block],
					frag_b.sub_d_frag[0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				mma_op(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
				mma_op(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_d_frag[bm + bk * num_m_block],
						frag_b.sub_frag  [bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
				mma_op(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_frag  [bm + bk * num_m_block],
						frag_b.sub_d_frag[bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk>
__device__ void mma_sync(
		fragment_f32<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_d,
		const fragment_f32<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_a,
		const fragment_f32<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>>& frag_b) {
	using Policy = mtk::wmma::detail::Policy<Op, mtk::wmma::op_with_error_correction, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;
	mtk::wmma::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy> zero_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			zero_op(frag_d.sub_frag[bm + bn * num_m_block]);
			mma_op(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_d.sub_frag[bm + bn * num_m_block]
					);
			zero_op(frag_d.sub_d_frag[bm + bn * num_m_block]);
			mma_op(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_d_frag[bm + 0  * num_m_block],
					frag_b.sub_frag  [0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			mma_op(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_frag  [bm + 0  * num_m_block],
					frag_b.sub_d_frag[0  + bn * num_k_block],
					frag_d.sub_d_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				mma_op(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
				mma_op(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_d_frag[bm + bk * num_m_block],
						frag_b.sub_frag  [bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
				mma_op(
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

#include "detail/no_cor.hpp"
