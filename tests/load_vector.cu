#include <iostream>
#include <type_traits>
#include <wmma_extension/hmma_f32_f32.hpp>

constexpr unsigned warp_size = 32;

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


__device__ half abs(const half a) {
	if (__half2float(a) < 0) {
		return -a;
	}
	return a;
}

/// Load

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void load_vector_ab_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr
		) {
	mtk::wmma::fragment_f32<Use, m, n, k, T, Layout> frag, frag_c;
	mtk::wmma::fill_fragment(frag, 0.0f);

	mtk::wmma::load_vector(frag, src_ptr);

	constexpr unsigned mem_m = mtk::wmma::detail::select_value<Use, m, k, m>();
	mtk::wmma::load_matrix_sync(frag_c, cor_ptr, mem_m);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(frag.x(i) - frag_c.x(i)));
	}
	for (unsigned i = 0; i < warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <int m, int n, int k, class T>
__global__ void load_vector_acc_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T> frag, frag_c;
	mtk::wmma::fill_fragment(frag, 0.0f);

	mtk::wmma::load_vector(frag, src_ptr, layout);

	constexpr unsigned mem_m = m;
	mtk::wmma::load_matrix_sync(frag_c, cor_ptr, mem_m, layout);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(frag.x(i) - frag_c.x(i)));
	}
	for (unsigned i = 0; i < warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
void load_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", to_string<Use>().c_str());
	std::printf("Layout : %s\n", to_string<Layout>().c_str());
	std::printf("Type   : %s\n", to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	constexpr unsigned mem_m = mtk::wmma::detail::select_value<Use, m, k, m>();
	constexpr unsigned mem_n = mtk::wmma::detail::select_value<Use, k, n, n>();

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	if constexpr (std::is_same<Use, nvcuda::wmma::accumulator>::value) {
		const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
		load_vector_acc_test_kernel<m, n, k, T><<<1, warp_size>>>(mat_mem, vec_mem, layout);
	} else {
		load_vector_ab_test_kernel<Use, m, n, k, T, Layout><<<1, warp_size>>>(mat_mem, vec_mem);
	}

	cudaDeviceSynchronize();

	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

/// Store

template <int m, int n, int k, class T>
__global__ void store_vector_acc_test_kernel(
		float* const dst_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, m, n, k, T> frag;

	constexpr unsigned mem_m = m;
	mtk::wmma::load_matrix_sync(frag, src_ptr, mem_m, layout);

	mtk::wmma::store_vector(dst_ptr, frag, layout);
}

template <class Use, int m, int n, int k, class T, class Layout>
void store_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", to_string<Use>().c_str());
	std::printf("Layout : %s\n", to_string<Layout>().c_str());
	std::printf("Type   : %s\n", to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	constexpr unsigned mem_m = mtk::wmma::detail::select_value<Use, m, k, m>();
	constexpr unsigned mem_n = mtk::wmma::detail::select_value<Use, k, n, n>();

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* res_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);
	cudaMallocHost(&res_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
	store_vector_acc_test_kernel<m, n, k, T><<<1, warp_size>>>(mat_mem, mat_mem, layout);

	cudaDeviceSynchronize();

	float max_error = 0.0f;
	for (unsigned i = 0; i < vec_len; i++) {
		const auto diff = mat_mem[i] - vec_mem[i];
		max_error = std::max(max_error, std::abs(diff));
	}
	std::printf("Error  : %e\n", max_error);

	cudaFree(res_mem);
	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

int main() {
	load_vector_test<nvcuda::wmma::matrix_a   , 32, 32, 32, half, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_b   , 32, 32, 32, half, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_a   , 32, 32, 32, half, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_b   , 32, 32, 32, half, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::row_major>();

	load_vector_test<nvcuda::wmma::matrix_a   , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_b   , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_a   , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_b   , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();

	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::col_major>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::row_major>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
}
